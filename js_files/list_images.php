<?php
$directory = 'userA_clothes/';
$images = glob($directory . "*.{jpg,jpeg,png,gif}", GLOB_BRACE);
echo json_encode(array_map('basename', $images));
?>
