run:
	DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker-compose build
	docker-compose run smoke
protos:
	source env.sh && \
	source activate $$CONDA_ENV && \
	cp realtery-protos/spec/service/vehicle_pose_detection_service.proto smoke/rpc/ && \
	python -m grpc_tools.protoc \
		-I . \
		--python_out=. \
		--grpc_python_out=. \
		smoke/rpc/*.proto