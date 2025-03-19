### How to run the project?

## First Run

1. git clone this repository

```bash
git clone <repo_url>
```

2. create virtual environment [python for windows powershell/cmd]. Then, install the libraries `pip install -r requirements.txt`

```bash
python3 -m venv <venv_name>
```

3. build the docker from docker-compose

```bash
docker-compose up -d
```

4. run each cells in `milvus-db.ipynb` to create the collection

5. run `main.py` to run the server

6. try call the server using call.py

## After First Run

Run this command for restart the docker:

```bash
docker-compose restart
```
