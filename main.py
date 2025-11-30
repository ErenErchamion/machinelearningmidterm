from pipeline.orchestrator import PipelineOrchestrator


def main():
    orchestrator = PipelineOrchestrator()
    orchestrator.run_full_pipeline()


if __name__ == "__main__":
    main()

