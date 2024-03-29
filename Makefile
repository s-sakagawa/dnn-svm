NVIDIA_SMI_PATH := $(shell which nvidia-smi)
IMAGE_NAME := dlsvm
CONTAINER_NAME := dlsvm
USER := dlsvm
WORKINGDIR := /var/www
PWD := $(shell pwd)

ifdef NVIDIA_SMI_PATH
    DOCKER_GPU_PARAMS := --gpus all
endif

.PHONY: _build/cpu
_build/cpu:
	@docker build --tag $(IMAGE_NAME) -f $(PWD)/docker/Dockerfile.cpu .

.PHONY: _build/gpu
_build/gpu:
	@docker build --tag $(IMAGE_NAME) -f $(PWD)/docker/Dockerfile.gpu .

.PHONY: build
build:
    ifdef NVIDIA_SMI_PATH
	    @$(MAKE) _build/gpu
    else
	    @$(MAKE) _build/cpu
    endif

.PHONY: run
run:
	@docker run \
		--rm -it \
		$(DOCKER_GPU_PARAMS) \
		--name $(CONTAINER_NAME) \
		--volume $(PWD):$(WORKINGDIR) \
		--publish 8888:8888 \
		--user $(USER) \
		$(IMAGE_NAME) \
		$(ARGS)

.PHONY: bash
bash: ARGS=bash
export ARGS
bash:
	@$(MAKE) run

.PHONY: lab
lab: ARGS=jupyter lab --ip=0.0.0.0
export ARGS
lab:
	@$(MAKE) run

.PHONY: test
test: ARGS=pytest tests
export ARGS
test:
	@$(MAKE) run

.PHONY: lint
lint: ARGS=flake8 src tests
export ARGS
lint:
	@$(MAKE) run
