const fs = require('fs');
// Load images from the "user images" folder
const imageContainer = document.querySelector('.image-container');

// Assuming you have a way to get the list of image file names
const imageFilenames = [''];
const folderPath = 'userA_clothes';

fs.readdir(folderPath, (err, files) => {
    if (err) {
        console.error('Error reading folder:', err);
        return;
    }

    console.log('Files in userA_clothes folder:');
    files.forEach(file => {
        console.log(file);
    });
});

imageFilenames.forEach(filename => {
    const img = document.createElement('img');
    img.src = `user-images/${filename}`;
    img.alt = '';
    imageContainer.appendChild(img);
});

// Add event listeners to buttons
const homeButton = document.querySelector('.home-button');
homeButton.addEventListener('click', () => {
    // Implement home button functionality
});

const settingsButton = document.querySelector('.settings-button');
settingsButton.addEventListener('click', () => {
    // Implement settings button functionality
});

const addButton = document.querySelector('.add-button');
addButton.addEventListener('click', () => {
    // Implement add more functionality
});
