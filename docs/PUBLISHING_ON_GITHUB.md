# Publishing this project on GitHub (step by step)

## 0. One-time setup (every team member)

```bash
git --version                                   # install from https://git-scm.com if missing
git config --global user.name  "Your Real Name"
git config --global user.email "the-email-used-on-your-GitHub-account"
```

> The name/e-mail you set here is what appears in `git log`. Checklist item 13 asks for the git log, so
> every member must commit **from their own configured identity** (never from someone else's laptop login).

## 1. Check that the project works before the first commit

```bash
cd sudoku_rescue_project
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python make_data.py            # downloads MNIST into data/raw (git-ignored)
python -m pytest -q            # all tests must pass (51 with the data present)
python run_demo.py             # check that outputs/demo/demo_overview.png looks right
```

## 2. Create the local repository (one person does this)

```bash
git init -b main
git add .
git status                      # CHECK: no data/raw, data/processed, .venv or __pycache__ listed
git commit -m "Initial commit: Sudoku rescue project (NN + search + CSP)"
```

## 3. Create the empty repository on GitHub

*Web:* github.com → **New repository** → name e.g. `handwritten-sudoku-solver` → choose **Private** while
you work (switch to Public after submission if the course allows it) → **do not** tick "Add a README /
.gitignore / license" (they already exist here) → **Create repository**.

*Or with the GitHub CLI:*
```bash
gh auth login
gh repo create handwritten-sudoku-solver --private --source=. --remote=origin --push
```

## 4. Connect and push (if you used the web page)

```bash
git remote add origin https://github.com/<your-user>/handwritten-sudoku-solver.git
git push -u origin main
```

GitHub no longer accepts account passwords for `git push`. Use either the GitHub CLI (`gh auth login`),
a **Personal Access Token** as the password (Settings → Developer settings → Tokens), or an SSH key and the
`git@github.com:<user>/<repo>.git` URL.

## 5. Add the team (Settings → Collaborators → Add people) and the lecturer, if required

Each teammate then runs `git clone <repo-url>` and repeats step 1 on their machine.

## 6. Team workflow that produces an honest, readable git log

```bash
git pull                                   # always start from the latest main
git switch -c feature/astar-heuristic      # one short-lived branch per task
# ...work, run tests...
git add sudoku_rescue/search/correction.py tests/test_search.py
git commit -m "search: prune A* states whose conflicts cannot be repaired"
git push -u origin feature/astar-heuristic
```
Open a **Pull Request** on GitHub, let a teammate review it, then *Squash and merge*.
The Actions tab runs `pytest` automatically on every push (`.github/workflows/tests.yml`).

Rules of thumb: small commits · one idea per commit · message = `module: what changed` · never commit
data, `.venv` or large binaries · never rewrite shared history (`push --force` on `main`).

## 7. What is (and is not) in the repository

| Tracked | Not tracked (git-ignored) |
|---|---|
| all source, tests, docs, `requirements.txt` | `data/raw/`, `data/processed/` (MNIST is re-downloaded by `make_data.py`) |
| `models/` (< 1 MB), `results/`, `figures/` (needed for the report) | `outputs/` (re-created by `run_demo.py`), `.venv/`, caches |

A fresh clone therefore reproduces everything with: `pip install -r requirements.txt` → `python make_data.py`
→ `python train_nn.py` → `python run_demo.py` → `python run_experiments.py`.

## 8. Before the deadline

```bash
python run_demo.py --misreads 2            # a SOLVED example makes the best README picture
cp outputs/demo/demo_overview.png docs/img/demo_overview.png
# add near the top of README.md:   ![demo](docs/img/demo_overview.png)

git log --pretty=format:'%h | %an | %ad | %s' --date=short > docs/git_log.txt    # checklist item 13
git shortlog -sn --all                      # commits per member (sanity check)

# put the final report PDF in docs/report/ and the slides in docs/slides/, then:
git add -A && git commit -m "docs: final report, slides, git log, contribution statements"
git tag -a v1.0-submission -m "Submitted version"
git push origin main --tags
```

Finally open the repository in a **private/incognito browser window** (or ask a teammate) and check that the
README renders, the figures show, and a fresh `git clone` + the commands in step 7 work.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `remote: Support for password authentication was removed` | use `gh auth login`, a token, or SSH (step 4) |
| `! [rejected] main -> main (fetch first)` | `git pull --rebase`, then push again |
| `data/raw/...` shows up in `git status` | `.gitignore` was added after the files were staged: `git rm -r --cached data/raw data/processed` |
| Wrong author name in `git log` | `git config user.name/email`, then `git commit --amend --reset-author` for the *last* commit only |
| Files over 100 MB rejected | you committed data by accident; remove it from the commit before pushing |
| CI fails on `pip install` | the pinned versions need Python ≥ 3.11 |
