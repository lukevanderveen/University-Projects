  import java.rmi.RemoteException;
  import java.rmi.registry.LocateRegistry;
  import java.rmi.registry.Registry;
  import java.util.Scanner;


  public class Client{

    private static Scanner scanner = new Scanner(System.in);

    private static void createAuction(Auction auction) throws RemoteException {
      AuctionSaleItem item = new AuctionSaleItem();
      item.name = "sample";
      item.description = "this is a sample item";
      item.reservePrice = 100;
      int userId = auction.register("user@example.com");
      int auctionId = auction.newAuction(userId, item);
      System.out.println("New auction created, ID = " + auctionId);
    }

    private static void listAuctions(Auction auction) throws Exception {
      AuctionItem[] items = auction.listItems();
      for (AuctionItem item: items){
        System.out.println("Item ID: " + item.itemID + ", Name: " + item.name + ", Highest Bid: " + item.highestBid);    }
    }

    private static void placeBid(Auction auction, int userId) throws Exception {
      try {
        listAuctions(auction);
        System.out.print("Enter item ID to bid on: ");
        int itemId = scanner.nextInt();
        System.out.print("Enter your bid amount: ");
        int bidAmount = scanner.nextInt();
        scanner.nextLine(); 

        //validation 
        if (auction.bid(userId, itemId, bidAmount) == true){
          System.out.println("UserID: " + userId + " has placed a bid on itemID: "+ itemId+ " for: £" + bidAmount);
        }else {
          System.out.println("bid was too low");
        }
      }catch (Exception e) {
        e.printStackTrace();
      }
    }

    private static void sellItem(Auction auction, int userId) {
      try {
          //System.out.println("Creating a new auction...");
          AuctionSaleItem item = new AuctionSaleItem();
          System.out.print("Enter item name: ");
          item.name = scanner.nextLine();
          System.out.print("Enter item description: ");
          item.description = scanner.nextLine();
          System.out.print("Enter reserve price: ");
          item.reservePrice = scanner.nextInt();
          scanner.nextLine(); // Consume newline

          int auctionId = auction.newAuction(userId, item);
          System.out.println("New auction created with ID: " + auctionId);

          // Close auction
          System.out.print("Enter auction ID to close: ");
          auctionId = scanner.nextInt();
          scanner.nextLine(); // Consume newline
          AuctionResult result = auction.closeAuction(userId, auctionId);
          if (result != null) {
              System.out.println("Auction closed. Winner: " + result.winningEmail + " with bid: £" + result.winningPrice);
          } else {
              System.out.println("Failed to close auction.");
          }
      } catch (Exception e) {
          e.printStackTrace();
      }
    }


    public static void main(String[] args) throws Exception {
      try{
        Registry reg = LocateRegistry.getRegistry("localhost");
        Auction auction = (Auction) reg.lookup("FrontEnd");
        if (auction == null){
          System.err.println("Could not connect to any primary replica. Exiting...");
          return;
        }
        createAuction(auction);
        int userID = auction.register("user@example.com");
        System.out.println("User registered with ID: " + userID);
        while (true){
          System.out.println("Select an option:");
          System.out.println("1. Sell an item");
          System.out.println("2. Buy an item");
          System.out.println("3. Exit");
          int choice = scanner.nextInt();
          scanner.nextLine();
          switch (choice) {
            case 1: // sell item
              sellItem(auction, userID);
              break;
            case 2: //buy item
              placeBid(auction, userID);
              break;
            case 3:
              System.out.println("Exiting...");
              return;
            default:
              System.out.println("Invalid option. Try again.");
              break;
          }
        }

      }catch (RemoteException e){
        System.err.println("Exception occurred during operation:");
        e.printStackTrace();
      }
    }
  }

  /* ================================================================= */
  /*
                            legacy code
  -- old main (requires a revamp)
  public static void main(String[] args) {
        if (args.length < 1) {
          System.out.println("Usage: java Client itemID");
          return;
        }
            int n = Integer.parseInt(args[0]);
            try {
            //server connection
              String name = "Auction";
              Registry registry = LocateRegistry.getRegistry("localhost");
              Auction auction = (Auction) registry.lookup(name);
              createAuction(auction);
              int user = auction.register("email@email.com");
              while (true) {
                //register user
                System.out.println("Select an option:");
                System.out.println("1. Sell an item");
                System.out.println("2. Buy an item");
                int choice = scanner.nextInt();
                scanner.nextLine();
                
                createAuction(auction);
                switch (choice) {
                  case 1: // sell item
                    sellItem(auction, user);
                    break;
                  case 2: //buy item
                    placeBid(auction, user);
                    break;
                }
              }
            } catch (Exception e) {
              System.err.println("Exception:");
              e.printStackTrace();
              }    
        }

            private static Auction connToPrimary() {
      String repPrefix = "Replica";
      int repID = 1; // sequentially search for primary (this is just a start index)
      while (true){
          try {
            String name = repPrefix + repID;
            Registry registry = LocateRegistry.getRegistry("localhost");
            Auction replica = (Auction) registry.lookup(name);

            //check for primary
            if (replica.getPrimaryReplicaID() != -1){
              System.out.println("Connected to primary replica: " + name);
              return replica;
            }
            repID++; // move to next replica
          }catch (Exception e){
            System.out.println("Failed to connect to replcia: "+ repPrefix + repID+ ". Assuming no more replicas");
            repID++; // move to next replica if there's a connection failure (primary failure detection)
          }
      }
    } 

  */
  /* ================================================================= */

