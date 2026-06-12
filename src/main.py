"""Main entry — run executive briefing with optional scheduler."""
import sys
from src.config import get_settings
from src.pipelines import get_context_from_pipelines, start_scheduler
from src.orchestrator import run_executive_briefing


def main() -> None:
    get_settings()
    # Load latest data (sync once if scheduler not used)
    context = get_context_from_pipelines()
    print("Running Chairman's Master Orchestrator (all agents -> executive briefing)...")
    print("\n" + "=" * 60 + "\nEXECUTIVE BRIEFING\n" + "=" * 60 + "\n")
    result = run_executive_briefing(context, stream_stdout=True)
    print("\n" + "=" * 60)
    return


if __name__ == "__main__":
    main()
    sys.exit(0)
