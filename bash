d# for installing required javascript packagaes
npx create-react-app glaucoma-detector
cd glaucoma-detector
npm install axios


# for building and running the docker file
docker build -t glaucoma-backend .
docker run -p 5000:5000 glaucoma-backend