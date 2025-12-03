<<<<<<< HEAD
public class GamingParty{
  private String theme;
  private BoardGame boardGame;
  private Player players[];
  private Snack snacks[];
  private Player winner;

  public GamingParty(String t, BoardGame b){
    theme = t;
    boardGame = b;
    players  = new Player[boardGame.getMaximumPlayers()];
    snacks = new Snack[(boardGame.getMaximumPlayers() * 2)];
    winner = null;
  }

  public void addPlayer(Player p){
    int counter = 0;
    //count amount of players in players[]
    for (int j = 0; j < boardGame.getMaximumPlayers(); j++){
      if(players[j] != null){
        counter++;
      }
    } 
    if(counter >= boardGame.getMaximumPlayers()){
        //if players[] maxed out
        System.out.println("The maximum number of players has been reached for the game at this party");
    }else if (p.getAge() <= boardGame.getMinimmumAge()){
        //if player too young
        System.out.println("The player does not meet the age requirements for the board game at this party");
    }else{
      //check if player is in the players[] array
      for (int i = 0; i < boardGame.getMaximumPlayers(); i++){
        if(players[i] == null){
          players[i] = p;
          i = boardGame.getMaximumPlayers();        
        }else if(players[i].getName() == p.getName()){
          //stop loop
          i = boardGame.getMaximumPlayers();
          System.out.println("The player is already playing the game at this party");
          //add player to next nearest space
        }
      }    
    }
  }
  
  

  public void getPlayers(){
    for (int i = 0; i < boardGame.getMaximumPlayers(); i++){
      System.out.println(players[i].getName());
    }
  }


  public void play(){
    int counter = 0;
    int result;
    for (int i = 0; i < boardGame.getMaximumPlayers(); i++){
      if(players[i] != null){
        counter++;
      }
    }
    if (counter >= boardGame.getMinumumPlayers()){
      System.out.println("Play!");
    }else{
      result = boardGame.getMinumumPlayers() - counter;
      System.out.println("You need "+ result + " more player(s)!");
    }
  }

  public void addSnack(Snack s){
    int counter = 0;
    for (int i = 0; i < (boardGame.getMaximumPlayers()*2); i++){
      if(snacks[i] != null){
        counter++;
      }
    }
    if (counter >= (boardGame.getMaximumPlayers()*2)){
      System.out.println("There are enough snacks!");
    }else{
      for (int j = 0; j < (boardGame.getMaximumPlayers()*2); j++){
        if(snacks[j] == null){
          snacks[j] = s;
          j = (boardGame.getMaximumPlayers()*2);
        }
      }
    }
  }

  public void getSnacks(){
    for (int i = 0; i < boardGame.getMaximumPlayers()*2; i++){
      System.out.println(snacks[i].getDescription());
    }
  }

  public double getPartyCost(){
    double total = 0;
    for (int i = 0; i < (boardGame.getMaximumPlayers()*2); i++){
      if(snacks[i] != null){
        total = total + snacks[i].getCost();
      }
    }
    return total;
  }  

  public void setWinner(Player p){
    if (winner == p){
      System.out.println("The winner has already been decided! it was"+ p);
    }else{
      for (int i = 0; i < boardGame.getMaximumPlayers(); i++){
        if(players[i] == p){
          winner = p;
        }else if (i == boardGame.getMaximumPlayers()){
          System.out.println("The player didn't even play the game so cannot win!");
        }
      }
    }
  }

  public String getWinner(){
    return winner.getName();
  }

  public void outputPartyDetails(){
    System.out.println("Theme: "+ theme);
    System.out.println("Board game: "+ boardGame.getTitle());
    System.out.println("Players:");
    for (int i = 0; i < boardGame.getMaximumPlayers(); i++){
      if (players[i] != null){
        System.out.println(players[i].getName());
      }
    } 
    System.out.println("Snacks:");
    for (int k = 0; k < boardGame.getMaximumPlayers()*2; k++){
      if (snacks[k] != null){
        System.out.println(snacks[k].getDescription()+ " provided by "+ snacks[k].getProvider());
      }
    }
    if (winner == null){
      System.out.println("No winner yet!");
    }else{
      System.out.println("The winner is "+winner.getName()+ "!");
    }
  }
  
  public void calculateRecommendedSnacks(){
    int count = 0;
    int totalSnacks = Math.round(boardGame.getDuration()/60);
    //maximum snacks 
    for (int i = 0; i < boardGame.getMaximumPlayers(); i++){
      if (players[i] != null){
        count++;
      }
    }
    if (totalSnacks > boardGame.getMaximumPlayers()*2){
      totalSnacks = boardGame.getMaximumPlayers()*2;
      System.out.println("A total of "+ totalSnacks +" snack(s) are recomended for the game");
    }else if (count == 0){
      System.out.println("A total of 0 snack(s) are recomended for the game");
      //minimum snacks
      totalSnacks = count;
      System.out.println("A total of "+ totalSnacks +" snack(s) are recomended for the game");
    }else if(totalSnacks >= 0){
      totalSnacks = count;
      System.out.println("A total of "+ totalSnacks +" snack(s) are recomended for the game");

    }else{
      System.out.println("A total of "+ totalSnacks +" snack(s) are recomended for the game");
    }
  }
}
=======
public class GamingParty{
  private String theme;
  private BoardGame boardGame;
  private Player players[];
  private Snack snacks[];
  private Player winner;

  public GamingParty(String t, BoardGame b){
    theme = t;
    boardGame = b;
    players  = new Player[boardGame.getMaximumPlayers()];
    snacks = new Snack[(boardGame.getMaximumPlayers() * 2)];
    winner = null;
  }

  public void addPlayer(Player p){
    int counter = 0;
    //count amount of players in players[]
    for (int j = 0; j < boardGame.getMaximumPlayers(); j++){
      if(players[j] != null){
        counter++;
      }
    } 
    if(counter >= boardGame.getMaximumPlayers()){
        //if players[] maxed out
        System.out.println("The maximum number of players has been reached for the game at this party");
    }else if (p.getAge() <= boardGame.getMinimmumAge()){
        //if player too young
        System.out.println("The player does not meet the age requirements for the board game at this party");
    }else{
      //check if player is in the players[] array
      for (int i = 0; i < boardGame.getMaximumPlayers(); i++){
        if(players[i] == null){
          players[i] = p;
          i = boardGame.getMaximumPlayers();        
        }else if(players[i].getName() == p.getName()){
          //stop loop
          i = boardGame.getMaximumPlayers();
          System.out.println("The player is already playing the game at this party");
          //add player to next nearest space
        }
      }    
    }
  }
  
  

  public void getPlayers(){
    for (int i = 0; i < boardGame.getMaximumPlayers(); i++){
      System.out.println(players[i].getName());
    }
  }


  public void play(){
    int counter = 0;
    int result;
    for (int i = 0; i < boardGame.getMaximumPlayers(); i++){
      if(players[i] != null){
        counter++;
      }
    }
    if (counter >= boardGame.getMinumumPlayers()){
      System.out.println("Play!");
    }else{
      result = boardGame.getMinumumPlayers() - counter;
      System.out.println("You need "+ result + " more player(s)!");
    }
  }

  public void addSnack(Snack s){
    int counter = 0;
    for (int i = 0; i < (boardGame.getMaximumPlayers()*2); i++){
      if(snacks[i] != null){
        counter++;
      }
    }
    if (counter >= (boardGame.getMaximumPlayers()*2)){
      System.out.println("There are enough snacks!");
    }else{
      for (int j = 0; j < (boardGame.getMaximumPlayers()*2); j++){
        if(snacks[j] == null){
          snacks[j] = s;
          j = (boardGame.getMaximumPlayers()*2);
        }
      }
    }
  }

  public void getSnacks(){
    for (int i = 0; i < boardGame.getMaximumPlayers()*2; i++){
      System.out.println(snacks[i].getDescription());
    }
  }

  public double getPartyCost(){
    double total = 0;
    for (int i = 0; i < (boardGame.getMaximumPlayers()*2); i++){
      if(snacks[i] != null){
        total = total + snacks[i].getCost();
      }
    }
    return total;
  }  

  public void setWinner(Player p){
    if (winner == p){
      System.out.println("The winner has already been decided! it was"+ p);
    }else{
      for (int i = 0; i < boardGame.getMaximumPlayers(); i++){
        if(players[i] == p){
          winner = p;
        }else if (i == boardGame.getMaximumPlayers()){
          System.out.println("The player didn't even play the game so cannot win!");
        }
      }
    }
  }

  public String getWinner(){
    return winner.getName();
  }

  public void outputPartyDetails(){
    System.out.println("Theme: "+ theme);
    System.out.println("Board game: "+ boardGame.getTitle());
    System.out.println("Players:");
    for (int i = 0; i < boardGame.getMaximumPlayers(); i++){
      if (players[i] != null){
        System.out.println(players[i].getName());
      }
    } 
    System.out.println("Snacks:");
    for (int k = 0; k < boardGame.getMaximumPlayers()*2; k++){
      if (snacks[k] != null){
        System.out.println(snacks[k].getDescription()+ " provided by "+ snacks[k].getProvider());
      }
    }
    if (winner == null){
      System.out.println("No winner yet!");
    }else{
      System.out.println("The winner is "+winner.getName()+ "!");
    }
  }
  
  public void calculateRecommendedSnacks(){
    int count = 0;
    int totalSnacks = Math.round(boardGame.getDuration()/60);
    //maximum snacks 
    for (int i = 0; i < boardGame.getMaximumPlayers(); i++){
      if (players[i] != null){
        count++;
      }
    }
    if (totalSnacks > boardGame.getMaximumPlayers()*2){
      totalSnacks = boardGame.getMaximumPlayers()*2;
      System.out.println("A total of "+ totalSnacks +" snack(s) are recomended for the game");
    }else if (count == 0){
      System.out.println("A total of 0 snack(s) are recomended for the game");
      //minimum snacks
      totalSnacks = count;
      System.out.println("A total of "+ totalSnacks +" snack(s) are recomended for the game");
    }else if(totalSnacks >= 0){
      totalSnacks = count;
      System.out.println("A total of "+ totalSnacks +" snack(s) are recomended for the game");

    }else{
      System.out.println("A total of "+ totalSnacks +" snack(s) are recomended for the game");
    }
  }
}
>>>>>>> 07f262e6dfdd43ebe007a76e1b5e3158a6049aec
