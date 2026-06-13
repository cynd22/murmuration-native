//! "Now playing" media source — MPRIS over D-Bus, like KDE's plasmusic-toolbar
//! but rendered by the app itself. A background thread polls the active player
//! ~4x/sec for metadata / position / play state and decodes album art; the UI
//! reads cheap snapshots and sends transport commands (play/pause, prev, next,
//! seek) back over a channel that the same thread executes on the live player.
//!
//! Resilient: every D-Bus / network / decode failure is logged and swallowed,
//! falling back to a default (not-playing) snapshot; if the player disappears
//! the finder is rebuilt and a fresh one is picked.
//!
//! The MPRIS polling + player scoring + art loading is the verified design from
//! the build agent; the command channel and transport control are added on top.

use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use mpris::{PlaybackStatus, PlayerFinder};

/// Decoded album art, ready to become an egui texture.
pub struct ArtImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>, // tightly packed RGBA8 (width*height*4)
}

/// Cheap-to-clone playback snapshot (no pixel data — see `take_new_art`).
#[derive(Clone, Default)]
pub struct NowPlaying {
    pub title: String,
    pub artist: String,
    pub album: String,
    pub position_secs: f64,
    pub length_secs: f64,
    pub playing: bool,
    pub art_url: String,
    pub player: String,
}

/// Transport commands sent from the UI to the poll thread.
pub enum Command {
    PlayPause,
    Next,
    Previous,
    /// Seek to an absolute position in seconds.
    SeekTo(f64),
}

#[derive(Default)]
struct Shared {
    np: NowPlaying,
    pending_art: Option<ArtImage>,
}

pub struct NowPlayingClient {
    shared: Arc<Mutex<Shared>>,
    cmd_tx: Sender<Command>,
}

impl NowPlayingClient {
    pub fn connect() -> Self {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let worker = shared.clone();
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        thread::Builder::new()
            .name("nowplaying-mpris".into())
            .spawn(move || poll_loop(worker, cmd_rx))
            .expect("spawn nowplaying thread");
        Self { shared, cmd_tx }
    }

    pub fn snapshot(&self) -> NowPlaying {
        match self.shared.lock() {
            Ok(g) => g.np.clone(),
            Err(p) => p.into_inner().np.clone(),
        }
    }

    /// Decoded art, returned exactly once after the track's art changes (so the
    /// UI uploads a texture only on track change).
    pub fn take_new_art(&self) -> Option<ArtImage> {
        match self.shared.lock() {
            Ok(mut g) => g.pending_art.take(),
            Err(p) => p.into_inner().pending_art.take(),
        }
    }

    pub fn send(&self, cmd: Command) {
        let _ = self.cmd_tx.send(cmd);
    }
}

fn poll_loop(shared: Arc<Mutex<Shared>>, cmd_rx: Receiver<Command>) {
    // recv_timeout doubles as the poll interval AND makes transport commands
    // fire instantly (we wake the moment one arrives, act, then re-poll).
    const POLL: Duration = Duration::from_millis(250);

    let mut loaded_art_url = String::new();
    let mut finder: Option<PlayerFinder> = None;

    loop {
        if finder.is_none() {
            match PlayerFinder::new() {
                Ok(f) => finder = Some(f),
                Err(e) => {
                    log::warn!("nowplaying: cannot connect to D-Bus: {e}");
                    publish_idle(&shared);
                    if wait_or_quit(&cmd_rx, POLL, None) {
                        return;
                    }
                    continue;
                }
            }
        }
        let f = finder.as_ref().unwrap();

        let player = pick_player(f);
        let Some(player) = player else {
            publish_idle(&shared);
            if wait_or_quit(&cmd_rx, POLL, None) {
                return;
            }
            continue;
        };

        let metadata = match player.get_metadata() {
            Ok(m) => m,
            Err(e) => {
                log::warn!("nowplaying: get_metadata failed (player gone?): {e}");
                finder = None;
                publish_idle(&shared);
                if wait_or_quit(&cmd_rx, POLL, None) {
                    return;
                }
                continue;
            }
        };

        let title = metadata.title().unwrap_or_default().to_string();
        let artist = metadata.artists().map(|a| a.join(", ")).unwrap_or_default();
        let album = metadata.album_name().unwrap_or_default().to_string();
        let art_url = metadata.art_url().unwrap_or_default().to_string();
        let length_secs = metadata.length().map(|d| d.as_secs_f64()).unwrap_or(0.0);
        let playing = matches!(player.get_playback_status(), Ok(PlaybackStatus::Playing));
        let position_secs = player.get_position().map(|d| d.as_secs_f64()).unwrap_or(0.0);
        let player_name = player.identity().to_string();

        let np = NowPlaying {
            title,
            artist,
            album,
            position_secs,
            length_secs,
            playing,
            art_url: art_url.clone(),
            player: player_name,
        };

        if art_url != loaded_art_url {
            loaded_art_url = art_url.clone();
            if art_url.is_empty() {
                if let Ok(mut g) = shared.lock() {
                    g.pending_art = None;
                }
            } else {
                match load_art(&art_url) {
                    Ok(img) => {
                        if let Ok(mut g) = shared.lock() {
                            g.pending_art = Some(img);
                        }
                    }
                    Err(e) => log::warn!("nowplaying: art load failed for {art_url}: {e}"),
                }
            }
        }

        if let Ok(mut g) = shared.lock() {
            g.np = np;
        }

        // Wait for the next poll OR a transport command; execute commands on
        // the live player immediately.
        if wait_or_quit(&cmd_rx, POLL, Some((&player, position_secs))) {
            return;
        }
    }
}

/// Block up to `dur` for a command. Returns true if the channel is disconnected
/// (UI dropped -> exit the thread). Executes any command that arrives.
fn wait_or_quit(
    cmd_rx: &Receiver<Command>,
    dur: Duration,
    player: Option<(&mpris::Player, f64)>,
) -> bool {
    match cmd_rx.recv_timeout(dur) {
        Ok(cmd) => {
            if let Some((p, pos)) = player {
                exec(p, pos, cmd);
            }
            false
        }
        Err(RecvTimeoutError::Timeout) => false,
        Err(RecvTimeoutError::Disconnected) => true,
    }
}

fn exec(player: &mpris::Player, position_secs: f64, cmd: Command) {
    let r = match cmd {
        Command::PlayPause => player.play_pause(),
        Command::Next => player.next(),
        Command::Previous => player.previous(),
        Command::SeekTo(target) => {
            // Relative seek by the delta (microseconds), the broadly-supported path.
            let offset = ((target - position_secs) * 1_000_000.0) as i64;
            player.seek(offset)
        }
    };
    if let Err(e) = r {
        log::warn!("nowplaying: transport command failed: {e}");
    }
}

/// Prefer a real music player with rich metadata over a browser tab that
/// happens to be "active". Falls back to find_active/find_first.
fn pick_player(finder: &PlayerFinder) -> Option<mpris::Player> {
    let players = match finder.find_all() {
        Ok(p) if !p.is_empty() => p,
        _ => return finder.find_active().or_else(|_| finder.find_first()).ok(),
    };

    let mut best: Option<(i32, mpris::Player)> = None;
    for player in players {
        let mut score = 0i32;
        let bus = player.bus_name().to_ascii_lowercase();
        const MUSIC_HINTS: [&str; 9] = [
            "spotify", "vlc", "mpv", "audacious", "rhythmbox", "clementine", "elisa", "cmus",
            "strawberry",
        ];
        if MUSIC_HINTS.iter().any(|h| bus.contains(h)) {
            score += 1000;
        }
        if let Ok(md) = player.get_metadata() {
            if md.art_url().map(|s| !s.is_empty()).unwrap_or(false) {
                score += 100;
            }
            if md.album_name().map(|s| !s.is_empty()).unwrap_or(false) {
                score += 50;
            }
            if md.title().map(|s| !s.is_empty()).unwrap_or(false) {
                score += 10;
            }
        }
        if matches!(player.get_playback_status(), Ok(PlaybackStatus::Playing)) {
            score += 5;
        }
        match &best {
            Some((bs, _)) if *bs >= score => {}
            _ => best = Some((score, player)),
        }
    }
    best.map(|(_, p)| p)
}

fn publish_idle(shared: &Arc<Mutex<Shared>>) {
    if let Ok(mut g) = shared.lock() {
        g.np = NowPlaying::default();
    }
}

fn load_art(art_url: &str) -> Result<ArtImage, String> {
    let bytes = if let Some(path) = art_url.strip_prefix("file://") {
        let path = path.find('/').map(|i| &path[i..]).unwrap_or(path);
        std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?
    } else if art_url.starts_with("http://") || art_url.starts_with("https://") {
        fetch_http(art_url)?
    } else {
        return Err(format!("unsupported art_url scheme: {art_url}"));
    };
    let dyn_img = image::load_from_memory(&bytes).map_err(|e| format!("decode: {e}"))?;
    let rgba = dyn_img.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(ArtImage {
        width,
        height,
        rgba: rgba.into_raw(),
    })
}

fn fetch_http(url: &str) -> Result<Vec<u8>, String> {
    let bytes = ureq::get(url)
        .call()
        .map_err(|e| format!("http get: {e}"))?
        .body_mut()
        .read_to_vec()
        .map_err(|e| format!("read body: {e}"))?;
    Ok(bytes)
}
