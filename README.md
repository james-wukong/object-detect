# object-detect
 
## How to use

1. install docker in your system
2. enter root directory of this project:
3. run following command in your terminal:

```sh
# linux -> linux/amd64, mac m1/m2 -> linux/arm64
docker buildx build --platform linux/arm64 -t object_detect .

docker run -d --name object_detect -p 8080:8080 object_detect
# docker run -d --name object_detect -p 8080:8080 -v ~/Documents/Docker-Volumns/object-detect:/app  object_detect

docker exec -it object_detect /bin/bash

uvincorn main:app --reload

```
