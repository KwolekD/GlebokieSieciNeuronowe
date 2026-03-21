build:
	docker build -t glebokie-sieci-neuronowe .

run:
	docker run --rm -it --gpus all -p 8888:8888 -v "$(PWD):/app" glebokie-sieci-neuronowe jupyter lab --ip=0.0.0.0 --allow-root --no-browser