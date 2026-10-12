REEL AUTOMATOR (Cloudflare R2 edition)

Files
  reel_automator_r2.py   the website/app (Streamlit)
  r2_storage.py          Cloudflare R2 helper (cannot delete; cannot write to the base bucket)
  check_r2.py            read-only test of your R2 settings
  requirements.txt       Python packages
  env.example.txt        template for your secret settings -> copy to a file named .env
  .gitignore             keeps .env out of GitHub

Quick start (Windows, VS Code terminal, inside this folder)
  1. Copy env.example.txt to a new file named exactly  .env  and fill in the values
  2. python -m venv .venv
  3. .venv\Scripts\activate
  4. pip install -r requirements.txt
  5. python check_r2.py
  6. streamlit run reel_automator_r2.py

Never share or upload the .env file.
