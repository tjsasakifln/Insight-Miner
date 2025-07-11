# Insight Miner

## Problem

Companies collect customer feedback from various sources, but extracting actionable insights from this data can be a slow and manual process. Manually analyzing thousands of reviews is impractical, and product and marketing teams miss the opportunity to quickly understand customer needs and frustrations.

## Solution

Insight Miner is an interactive dashboard that automates the analysis of customer feedback. It uses natural language processing (NLP) to identify sentiment trends, extract key discussion topics, and visualize the data in a clear and concise way. With Insight Miner, companies can:

*   **Centralize feedback:** Upload a CSV file with your customer reviews.
*   **Analyze sentiment:** Quickly understand if customers are satisfied or dissatisfied.
*   **Extract topics:** Discover the main topics of conversation for your customers.
*   **Visualize data:** Explore data through interactive charts and word clouds.
*   **Get AI-powered summaries:** Generate concise summaries of insights with the power of GPT-4.

## How to Run Locally

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/tjsasakifln/Insight-Miner.git
    cd Insight-Miner
    ```

2.  **Create and activate the virtual environment:**

    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file** in the project root and add your OpenAI API key:

    ```
    OPENAI_API_KEY=your-key-here
    ```

5.  **Run the application:**

    ```bash
    streamlit run scripts/app.py
    ```

## Future Integrations

*   **Google NLP:** Integration with the Google Natural Language API for more advanced sentiment analysis and entity extraction.
*   **Amazon Comprehend:** Use of Amazon Comprehend for large-scale text analysis.

---

**Want this running in your company? Talk to me.**