
#to build
#docker compose --profile gpu up -d --build

# copy env file
cp .env.hug .env


docker compose --profile gpu up -d 
