//! Shader hot-reload: watch shaders/ and report when anything changes.
//! The actual pipeline rebuild lives with the pipelines (sim.rs / render.rs);
//! this module only does file watching + source loading.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use notify::{RecursiveMode, Watcher};

pub struct Sources {
    pub common: String,
    pub grid: String,
    pub velocity: String,
    pub position: String,
    pub bird: String,
}

impl Sources {
    pub fn load(dir: &Path) -> std::io::Result<Self> {
        let read = |name: &str| std::fs::read_to_string(dir.join(name));
        Ok(Self {
            common: read("common.wgsl")?,
            grid: read("grid.wgsl")?,
            velocity: read("velocity.wgsl")?,
            position: read("position.wgsl")?,
            bird: read("bird.wgsl")?,
        })
    }

    /// Compute shaders get the shared Params prelude prepended.
    pub fn with_common(&self, body: &str) -> String {
        format!("{}\n{}", self.common, body)
    }
}

pub struct ShaderWatcher {
    rx: mpsc::Receiver<notify::Result<notify::Event>>,
    _watcher: notify::RecommendedWatcher,
    pending_since: Option<Instant>,
}

impl ShaderWatcher {
    pub fn new(dir: PathBuf) -> notify::Result<Self> {
        let (tx, rx) = mpsc::channel();
        let mut watcher = notify::recommended_watcher(tx)?;
        // Watch the directory, not files — editors atomic-save via rename,
        // which would silently kill a per-file watch.
        watcher.watch(&dir, RecursiveMode::Recursive)?;
        Ok(Self {
            rx,
            _watcher: watcher,
            pending_since: None,
        })
    }

    /// True once changes have settled for a debounce window (editors emit
    /// several events per save; partially-written files fail to parse).
    pub fn take_changed(&mut self) -> bool {
        while let Ok(event) = self.rx.try_recv() {
            if let Ok(e) = event {
                // Writes only — our own reload READS the files, and counting
                // those access events would make the reload retrigger itself
                // forever.
                let is_write = matches!(
                    e.kind,
                    notify::EventKind::Modify(_)
                        | notify::EventKind::Create(_)
                        | notify::EventKind::Remove(_)
                );
                if is_write
                    && e.paths
                        .iter()
                        .any(|p| p.extension().is_some_and(|x| x == "wgsl"))
                {
                    self.pending_since = Some(Instant::now());
                }
            }
        }
        if let Some(t) = self.pending_since {
            if t.elapsed() > Duration::from_millis(60) {
                self.pending_since = None;
                return true;
            }
        }
        false
    }
}
