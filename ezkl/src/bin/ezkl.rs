//! Dark‑EZKL `ezkl` CLI binary.
//!
//! This binary delegates all command behavior to the library implementation
//! (`ezkl::execute` + `ezkl::commands`).
//!
//! The previous version of this file was a *stub* that only wrote placeholder
//! artifacts (e.g. `proof.json = "dummy"`), which could make pipelines appear to
//! succeed while doing no cryptography.

#[cfg(not(feature = "ezkl"))]
fn main() {
    eprintln!(
        "The `ezkl` binary was built without the `ezkl` feature enabled.\n\
         This build cannot compile circuits / generate proofs.\n\
         \n\
         Rebuild with:\n\
         \n\
         \tcargo install --path ezkl --features ezkl\n"
    );
    std::process::exit(1);
}

#[cfg(feature = "ezkl")]
mod real_cli {
    use clap::Parser;
    use std::process::ExitCode;

    use ezkl::{commands::Commands, execute};

    #[derive(Debug, Parser)]
    #[command(
        name = "ezkl",
        about = "Dark‑EZKL (full CLI)",
        version,
        arg_required_else_help = true
    )]
    struct Cli {
        #[command(subcommand)]
        command: Commands,
    }

    pub async fn run() -> ExitCode {
        let cli = Cli::parse();

        match execute::run(cli.command).await {
            Ok(output) => {
                let out = output.trim();
                if !out.is_empty() {
                    println!("{out}");
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::from(1)
            }
        }
    }
}

#[cfg(feature = "ezkl")]
#[tokio::main]
async fn main() -> std::process::ExitCode {
    real_cli::run().await
}
