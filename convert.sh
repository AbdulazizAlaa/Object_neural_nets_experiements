for name in test1/*.jpg; do
  echo $name
  if [[ $name == *'cat'* ]]; then
    echo "cat"
    cp /home/abdulaziz/Downloads/datasets/cats_dogs_classifier/$name /home/abdulaziz/Downloads/datasets/cats_dogs_classifier/data/test/cats
  fi
  if [[ $name == *'dog'* ]]; then
    echo "dog"
    cp /home/abdulaziz/Downloads/datasets/cats_dogs_classifier/$name /home/abdulaziz/Downloads/datasets/cats_dogs_classifier/data/test/dogs
  fi
  # convert $name -resize 150x150\! $name
done
