document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();
    var username = document.getElementById('username').value;
    var password = document.getElementById('password').value;

    // You can perform your login logic here
    // For demo purposes, let's just log the username and password
    console.log('Username: ' + username);
    console.log('Password: ' + password);

    // You can redirect the user to another page upon successful login
    // window.location.href = 'dashboard.html';
});
