import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class GenerateCSV {
static String[] names = {
        "Elle Hale", "Isabella Rodriguez", "Ellie Davis", "Nolan Mccullough", "Jaycee Espinoza",
        "Tiara Rivera", "Grady Golden", "Kenyon Schwartz", "Brett Harrington", "Gunnar Hurley",
        "Chase Weeks", "Angel Summers", "Tripp Moore", "Kendra Nicholson", "Henry Tucker",
        "Clark Lutz", "Dashawn Reyes", "Sidney Pearson", "Aleena Guerra", "Tia Luna", "Victoria Barnes",
        "Justus Rollins", "Rory Irwin", "Hayden Hobbs", "Immanuel Gomez", "Mylie Peck", "Kamila Guerrero",
        "Frances Page", "Kaitlyn Schaefer", "Nina Lucero", "Marco Navarro", "Hadassah Hess", "Braylen Case",
        "Ethen Patel", "Jaxson Landry", "Iliana Pham", "Salvador Cruz", "Kathryn Stout", "Gary Blankenship",
        "Leah Armstrong", "Arturo Pacheco", "Dallas Humphrey", "Julianna Kirk", "Philip Barajas",
        "Kobe Hicks", "Aden Conrad", "Emerson Hebert", "Jacquelyn Diaz", "Leah Harding", "Braelyn Knapp",
        "Christopher Norris", "Jade Leon", "Rylee Wheeler", "Patricia Morris", "Kadin Nguyen",
        "Gisselle Escobar", "Penelope Valdez", "Gianni Townsend", "Emilee Lucas", "Kendall Conner",
        "Natalee Pena", "Eleanor Shannon", "Karli Schmidt", "Keaton Rivera", "Vivian Edwards", "Rodolfo Cuevas",
        "Addison Rocha", "Rory Wall", "Kiana Hayes", "Alonzo Butler", "Russell Boyle", "Brody Mack",
        "Gaige Thomas", "Braden Grant", "Mara Contreras", "Savanna Hernandez", "Briana Horn", "Josh Tanner",
        "Hassan Robertson", "Rhett Gallegos", "Pierre Lester", "Landyn Ramos", "Gauge Morris",
        "Winston Buchanan", "Nicolas Hess", "Octavio Hale", "Carsen Vincent", "Leland Schultz",
        "Skylar Clayton", "Dexter Craig", "Jamar Cook", "Chace Boyd", "Ethen Duncan", "Logan Adams",
        "Genesis Humphrey", "Mattie Mcclure", "Randall Phelps", "Holly Bolton", "Jaylene Lozano",
        "Emiliano Norman", "Van Baker", "Kristen Sims", "Charles Cameron", "Jaelynn Watkins", "Hana Mercer",
        "Audrey Ramos", "Zion Castaneda", "Trace Haney", "Zaiden Mejia", "Curtis Nash", "Emerson Liu",
        "Logan Barnett", "Jaelyn Clay", "Camila Gordon", "Cameron Cobb", "Jakobe Mcconnell", "Thalia Macias",
        "Ryder Cross", "Jessica Carney", "Amina Sparks", "Janelle Mcgee", "Susan Barker", "Kiley Knapp",
        "Roman Ashley", "Kaiden Watson", "Grayson Heath", "Jayvon Stephens", "Joey Morgan", "Karina Kaiser",
        "Jacqueline Perez", "Fabian Richard", "Cruz Rivers", "Landin Nielsen", "Harley Juarez", "Maleah Lewis",
        "Deandre Chandler", "Deshawn Barker", "Glenn Zavala", "Finley Brooks", "Travis Stephenson",
        "Jeremy Finley", "Monica Caldwell", "Trystan Jacobs", "Jovanny Morton", "Dane Benson", "Cali Lynch",
        "Camryn Rosales", "Carmelo Santana", "Lyric Kirk", "Adalynn Bowen", "Ellis Hernandez", "Isis Medina",
        "Jesus Buckley", "Aryana Davila", "Deangelo Macdonald", "Kaylin Hart", "Braeden Mcgrath",
        "Kiersten Peters", "Cindy Flores", "Reese Werner", "Nico Murray", "Brennan Zimmerman", "Ethan Arnold",
        "Houston Kirby", "Giana Patrick", "Kendal Liu", "Colton Deleon", "Valentin Riley", "Camryn Davis",
        "Christopher Potts", "Martin Blankenship", "Livia George", "Isaiah Durham", "Landon Pugh",
        "Alexa Hess", "Rogelio Oconnell", "Brisa King", "Grayson Haynes", "Greta Moore", "Laci Boyd",
        "Whitney Moreno", "Amari Moreno", "Rowan Lester", "Ryan Farmer", "Emilia Friedman", "Antoine Werner",
        "Henry Mata", "Van Hernandez", "Jayleen Parsons", "Glenn Livingston", "Cesar Holland", "Nicholas Riddle",
        "Aliza Crosby", "Jake Robbins", "Elias Wood", "Nolan Campos", "Alexander Pollard", "Reed David",
        "Cason Austin", "Tommy Oconnell"
    };

    public static void main(String[] args) {
        generateCombinedCSV(names);
    }

    private static void generateCombinedCSV(String[] names) {
        
        try (FileWriter writer = new FileWriter("38862905.csv")) {
            for (int i = 1; i <= 200; i++) {
                String type = getRandomType();
                String data1 = getRandomData(type, names);
                writer.append(String.format("%d,%s,%s\n", i, type, data1));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static String getRandomType() {
        String[] types = {"Route", "Climber", "Route_Setter"};
        Random random = new Random();
        return types[random.nextInt(types.length)];
    }

    private static String getRandomData(String type, String[] names) {
        Random random = new Random();
        switch (type) {
            case "Route":
                return getRandomRouteData(random);
            case "Climber":
                return getRandomClimberData(random, names);
            case "Route_Setter":
                return getRandomRouteSetterData(random, names);
            default:
                return "";
        }
    }

    private static String getRandomRouteData(Random random) {
        int grade = random.nextInt(15);
        String type = getRandomCertificationLevel(random);
        String colour;
        if (grade == 0){
             colour = "Red";
        }else if (grade > 0 && grade <= 2){
             colour = "Blue";
        }else if (grade > 2 && grade <= 4){
             colour = "Yellow";
        }else if (grade > 4 && grade <= 6){
             colour = "White";
        }else if (grade > 6 && grade <= 8){
             colour = "Black";
        }else if (grade > 8 && grade <= 10){
             colour = "Cyan";
        }else if (grade > 10 && grade <= 12){
             colour = "Grey";
        }else{
             colour = "Purple";
        }
        return String.format("%s,%d,%s", colour, grade, type);
    }

    private static String getRandomClimberData(Random random, String[] names) {
        int randomIndex = random.nextInt(names.length);
        String[] temp = names[randomIndex].split(" ");
        String forename = temp[0];
        String surname = temp[1];
        String dob = generateRandomDate(1970, 2015);
        String startDate = generateRandomDate(2010, 2024);
        String certificationLevel = getRandomCertificationLevel(random);
        boolean membership = random.nextBoolean();
        return String.format("%s,%s,%s,%s,%s,%s", forename, surname, dob, startDate, certificationLevel, membership);
    }

    private static String getRandomRouteSetterData(Random random, String[] names) {
        int randomIndex = random.nextInt(names.length);
        String[] temp = names[randomIndex].split(" ");
        String forename = temp[0];
        String surname = temp[1];
        String dob = generateRandomDate(1970, 2000);
        String startDate = generateRandomDate(2010, 2022);
        String employeeStatus = getRandomEmployeeStatus(random);
        return String.format("%s,%s,%s,%s,%s", forename, surname, dob, startDate, employeeStatus);
    }

    private static String generateRandomDate(int startYear, int endYear) {
        int year = (int) (Math.random() * (endYear - startYear + 1)) + startYear;
        int month = (int) (Math.random() * 12) + 1;
        int day = (int) (Math.random() * 28) + 1; // Just to keep it simple
        return String.format("%d-%02d-%02d", year, month, day);
    }

    private static String getRandomCertificationLevel(Random random) {
        String[] levels = {"Boulder", "Toperope", "Lead"};
        return levels[random.nextInt(levels.length)];
    }

    private static String getRandomEmployeeStatus(Random random) {
        String[] statuses = {"Full-time", "Part-time", "Manager", "Instructor"};
        return statuses[random.nextInt(statuses.length)];
    }
}
