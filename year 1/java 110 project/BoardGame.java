<<<<<<< HEAD
public class BoardGame{
  private String title;
  private String owner;
  private int minimumAge;
  private int duration;
  private int minimumPlayers;
  private int maximumPlayers;
  private double rating;

  public BoardGame(String t, Player o, int m, int d, int minP, int maxP, double r){
    title = t;
    owner = o.getName();
    minimumAge = m;
    duration = d;
    minimumPlayers = minP;
    maximumPlayers = maxP;
    if (r < 0 || r > 100){
      System.out.println("Raiting percentage not valid. Setting to 0.");
      r = 0;
    }else{
      rating = r;
    }
  }

  public String getTitle(){
    return title;
  }

  public String getOwner(){
    return owner;
  }

  public int getMinimmumAge(){
    return minimumAge;
  }

  public int getDuration(){
    return duration;
  }

  public int getMinumumPlayers(){
    return minimumPlayers;
  }

  public int getMaximumPlayers(){
    return maximumPlayers;
  }

  public double getRating(){
    return rating;
  }

  public void setRating(double r){
    if (r < 0 || r > 100){
      System.out.println("Raiting percentage not valid.");
    }else{
      rating = r;
      System.out.println("Raiting percentage valid");
    }
  }
}
=======
public class BoardGame{
  private String title;
  private String owner;
  private int minimumAge;
  private int duration;
  private int minimumPlayers;
  private int maximumPlayers;
  private double rating;

  public BoardGame(String t, Player o, int m, int d, int minP, int maxP, double r){
    title = t;
    owner = o.getName();
    minimumAge = m;
    duration = d;
    minimumPlayers = minP;
    maximumPlayers = maxP;
    if (r < 0 || r > 100){
      System.out.println("Raiting percentage not valid. Setting to 0.");
      r = 0;
    }else{
      rating = r;
    }
  }

  public String getTitle(){
    return title;
  }

  public String getOwner(){
    return owner;
  }

  public int getMinimmumAge(){
    return minimumAge;
  }

  public int getDuration(){
    return duration;
  }

  public int getMinumumPlayers(){
    return minimumPlayers;
  }

  public int getMaximumPlayers(){
    return maximumPlayers;
  }

  public double getRating(){
    return rating;
  }

  public void setRating(double r){
    if (r < 0 || r > 100){
      System.out.println("Raiting percentage not valid.");
    }else{
      rating = r;
      System.out.println("Raiting percentage valid");
    }
  }
}
>>>>>>> 07f262e6dfdd43ebe007a76e1b5e3158a6049aec
