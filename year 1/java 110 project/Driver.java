<<<<<<< HEAD
public class Driver {
    public static void main(String[] args){
        Player kit = new Player("Kit", 18);
        Player riley = new Player("Riley", 19);
        Snack chips = new Snack("Chips", 5, riley);
        BoardGame monopoly = new BoardGame("Monopoly", kit, 8, 120, 6, 0, 54.7);
        GamingParty birthday = new GamingParty("Birthday", monopoly);

        birthday.addPlayer(kit);
        birthday.addPlayer(riley);
        birthday.addSnack(chips);
        birthday.outputPartyDetails();
        birthday.calculateRecommendedSnacks();
    }       
}
=======
public class Driver {
    public static void main(String[] args){
        Player kit = new Player("Kit", 18);
        Player riley = new Player("Riley", 19);
        Snack chips = new Snack("Chips", 5, riley);
        BoardGame monopoly = new BoardGame("Monopoly", kit, 8, 120, 6, 0, 54.7);
        GamingParty birthday = new GamingParty("Birthday", monopoly);

        birthday.addPlayer(kit);
        birthday.addPlayer(riley);
        birthday.addSnack(chips);
        birthday.outputPartyDetails();
        birthday.calculateRecommendedSnacks();
    }       
}
>>>>>>> 07f262e6dfdd43ebe007a76e1b5e3158a6049aec
 