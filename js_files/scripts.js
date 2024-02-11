document.addEventListener('DOMContentLoaded', function() {
    var gallery = document.querySelector('.gallery');
    var basePath = 'userA_clothes/';
  
    // Fetch the list of images from the server-side script
    fetch('list_images.php') // Assuming 'list_images.php' outputs the list of image URLs
      .then(response => response.json()) // Expecting the server to return a JSON array
      .then(images => {
        // Add images to gallery
        images.forEach(function(imageSrc) {
          var img = document.createElement('img');
          img.src = basePath + imageSrc;
          img.alt = 'User Image';
          gallery.appendChild(img);
        });
      })
      .catch(error => {
        console.error('Error fetching images:', error);
      });
    
    var addClozyButton = document.getElementById('addClozy');
    addClozyButton.addEventListener('click', function() {
      console.log('Add new item functionality here');
      // Implement the functionality to add new items
    });
  });
  