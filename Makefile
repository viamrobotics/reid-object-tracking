test:
	python3 -m pytest tests/* -v --tb=line --showlocals --log-format="%(asctime)s %(levelname)s %(message)s" \
        --log-date-format="%Y-%m-%d %H:%M:%S"

debug:
	rm results2/cost_matrix.log
	python3 -m pytest tests/* -v --tb=line --showlocals --log-format="%(asctime)s %(levelname)s %(message)s" \
        --log-date-format="%Y-%m-%d %H:%M:%S"
	open results2/cost_matrix.log

lint:
	python3 -m pylint src --disable=W0719,R0902,R0913,R0914,R0903

executable:
	python -m PyInstaller --onefile --hidden-import="googleapiclient" src/main.py

dist/archive.tar.gz: meta.json dist/main dist/first_run.sh
	tar -czvf $@ $^
