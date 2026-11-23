# AITY
AITY stands for **AI Agent for Text Analysis**.
It is an R&D Project directed and produced by Dorina Rajanen and Mikko Rajanen at University of Oulu, INTERACT Research Group, UX Lab.

## Contributors:
Sultan Alshehry, Ngbede Linus, Timo Pihlava, Elina Ronkainen, Vilma Santamäki, Jessica Sivonen (2026 Bachelor’s Project course)

## Project Description

AITY – AI Agent for text analysis. AITY is a concept or project of automatic text analysis created by Dorina Rajanen. AITY builds on the ability of an AI agent that analyses a text document (research article or new article) and identifies keywords, themes, topics. AITY then extracts these categories and stores them in a database. AITY also compares different documents and creates different analyses on the documents. Course project assignment includes the following tasks (selection of these can be discussed with the team):
-	Design, prototype and implement an AI agent for text analysis together with the documentation and literature review
-	Choose the AI platform suitable for the task and document the choice made
-	Document the ethical aspects of using research articles or news articles by the AI agent based on literature and available sources of the articles (e.g., news sites or scholarly databases or forums)
-	Document and calculate sustainability indicators such as carbon footprint for each document analysed
-	Design and implement a simple user interface of the agent
-	Document implementation options local versus public agent
-	Experimental results on a given dataset

## Repository layout

```
src/
├── aity_desktop/   # the original Tkinter desktop app (Gemini + BERTs modes) — frozen, bug fixes only
└── aity_web/       # the new multi-user web service (FastAPI + Vue 3) — in development
web_ui/             # web service backend (FastAPI) and frontend (Vue 3) — in development
tests/
```

`aity_desktop` is kept fully runnable but frozen until a further decision. The web service is built separately in `aity_web` and never imports `aity_desktop`. See `src/aity_desktop/README.md`.

## How to use (desktop app)

1. Download the repository:

```
git clone https://github.com/UXLAB-OULU/AITY-AI-Agent-for-Text-Analysis-
```


2. Enter the source directory:

```
cd AITY-AI-Agent-for-text-analysis-
cd src
```


3. Create a python virtual environment

Windows:

```
python -m venv venv
```

MacOs/Linux:

```
python3 -m venv venv

```

4. Activate the virtual environment

Windows: 

```
venv\Scripts\activate.bat

```

MacOs/Linux:

```
source venv/bin/activate
```

5. Download dependencies

```
pip install -r aity_desktop/requirements.txt

```

6. Run the program

```
python -m aity_desktop

```

7. Enjoy!
