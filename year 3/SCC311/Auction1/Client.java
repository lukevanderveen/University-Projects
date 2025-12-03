import javax.crypto.Cipher;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;
import javax.crypto.SealedObject;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.util.Scanner;

/*
 * decrypt sealedObject then print
 */

public class Client{

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

  private static Scanner scanner = new Scanner(System.in);


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
            // implement some logical validation
            //create auction if it doesn't exist
            //if it does bid in that auction 
            //list items and placebids
            //close auction once auction has completed
            // want a selling client and a buying client
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
    }
/*
 * legacy code:
 //load key from file
  SecretKey secretKey = loadKey();
  if (secretKey == null) {
    System.out.println("Error: Key not found");
    return;
  }

  
Cipher cipher = Cipher.getInstance("AES");
cipher.init(Cipher.DECRYPT_MODE, secretKey);
AuctionItem item = (AuctionItem) result.getObject(cipher);

//File kFile = new File("keys/testKey.aes");
  private static SecretKey loadKey() throws Exception {
    File kFile = new File("keys/testKey.aes");
        if (kFile.exists()) {
            byte[] data = new byte[16]; 
            try (FileInputStream f = new FileInputStream(kFile)) {
                int bytesRead = f.read(data);
                if (bytesRead != 16) {
                    throw new IOException("Key file does not contain the correct number of bytes.");
                }
                return new SecretKeySpec(data, "AES");
            }
        }
        return null; 
    }

 */