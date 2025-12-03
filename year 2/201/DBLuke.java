import java.sql.*;

public class DBLuke {

    public static void main(String args[]){
        String JBDCurl = "jdbc:mysql://localhost:3306";
        String DBname = "CLIMBING";
        try (
            //Class.forName("com.mysql.jdbc.Driver");
            Connection connection = DriverManager.getConnection(JBDCurl);
            Statement statement = connection.createStatement();
        ){
            String createDB = "CREATE DATABASE IF NOT EXISTS" + DBname;
            statement.executeUpdate(createDB);
            System.out.println("Database "+ DBname + " successfully created");
        }catch(SQLException ex){
            ex.printStackTrace();
        }
    }
}   