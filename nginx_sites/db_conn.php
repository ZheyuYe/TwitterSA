<?php
include 'key.php';
header('Content-Type:application/json');//这个类型声明非常关键

function db_connect(){
    global $servername;
    global $username;
    global $password;
    global $dbname;
    try {
        $conn = new PDO("mysql:host=$servername;dbname=$dbname;charset=utf8mb4", $username, $password);
        // echo "Connection Successful<br>";
    }
    catch(PDOException $e)
    {
        echo 'error'.$e->getMessage();
        exit;
    }

    if (!$conn){
        echo "fail to connect mysql database";
        return false;
    }
    return $conn;
}

function db_select($conn,$query){
    try{
        $res=$conn->prepare($query);
        $res->execute();
    }
    catch(PDOException $e){
        // echo "Error: " . $e->getMessage();
        return false;
    }
    $data=$res->fetch(PDO::FETCH_ASSOC);
    return $data;
}
?>
