from pipeline import run_pipeline

if __name__ == "__main__":
    prompt = "A 3-scene documentary about Ayatollah Khomeini of Iran. Start immediately by showing his iconic face."
    print(f"Testing pipeline with prompt: '{prompt}'")
    final_video = run_pipeline(prompt, do_research=False, do_web_search=True)
    if final_video:
        print(f"Pipeline SUCCESS! Final Video: {final_video}")
    else:
        print("Pipeline FAILED.")
