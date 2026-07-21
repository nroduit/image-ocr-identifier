
#to build
#docker compose --profile gpu up -d --build

# copy env file
cp env_hug.txt .env


docker compose --profile gpu up -d 
