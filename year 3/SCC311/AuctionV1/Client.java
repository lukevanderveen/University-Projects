import javax.crypto.Cipher;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;
import javax.crypto.SealedObject;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.security.KeyFactory;
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.Signature;
import java.security.spec.X509EncodedKeySpec;
import java.util.Scanner;

/*
 * decrypt sealedObject then print
 */

public class Client{

  private static PublicKey serverPublicKey;
  private static PrivateKey clientPrivateKey;
  private static PublicKey clientPublicKey;
  private static String authToken;
  private static int userId;


  private static void authenticate(Auction auction) throws Exception {
    String clientChal = "ClientChallenge" + System.currentTimeMillis();
    ChallengeInfo chalInfo = auction.challenge(userId, clientChal);

    Signature sig = Signature.getInstance("SHA256withRSA");
    sig.initVerify(serverPublicKey);
    sig.update(clientChal.getBytes());
    if (!sig.verify(chalInfo.response)) {
        throw new SecurityException("Server challenge verification failed.");
    }

    sig.initSign(clientPrivateKey);
    sig.update(chalInfo.serverChallenge.getBytes());
    byte[] sigChallenge = sig.sign();

    TokenInfo tokenI = auction.authenticate(userId, sigChallenge);
    if (tokenI == null) {
        throw new SecurityException("Authentication failed.");
    }
    authToken = tokenI.token;
    System.out.println("Authenticated successfully. Token: " + authToken);
  }

    private static KeyPair generateKeyPair() throws NoSuchAlgorithmException {
      KeyPairGenerator keyGen = KeyPairGenerator.getInstance("RSA");
      keyGen.initialize(2048);
      return keyGen.generateKeyPair();
    }

  private static PublicKey loadPublicKey(String filePath) throws Exception {
      byte[] keyBytes = Files.readAllBytes(Paths.get(filePath));
      X509EncodedKeySpec spec = new X509EncodedKeySpec(keyBytes);
      KeyFactory keyFactory = KeyFactory.getInstance("RSA");
      return keyFactory.generatePublic(spec);
  }

  private static int createAuction(Auction auction, int userId) throws RemoteException {
    AuctionSaleItem item = new AuctionSaleItem();
    item.name = "sample";
    item.description = "this is a sample item";
    item.reservePrice = 100;
    //int userId = auction.register("user@example.com", clientPublicKey);
    System.out.println("userid: "+ userId+ " item: "+ item.name+ " authToken: "+authToken);
    int auctionId = auction.newAuction(userId, item, authToken);
    System.out.println("New auction created, ID = " + auctionId);
    return auctionId;
  }

  private static void listAuction(Auction auction) throws Exception {
    AuctionItem[] items = auction.listItems(userId, authToken);
    if (items != null){
      for (AuctionItem item: items){
       System.out.println("Item ID: " + item.itemID + ", Name: " + item.name + ", Highest Bid: " + item.highestBid);    
      }
    }else {
      System.out.println("no items available");
    }   
  }

  private static void placeBid(Auction auction, int userId) throws Exception {
    try {
      listAuction(auction);
      System.out.print("Enter item ID to bid on: ");
      int itemId = scanner.nextInt();
      System.out.print("Enter your bid amount: ");
      int bidAmount = scanner.nextInt();
      scanner.nextLine(); 
        //validation 
      if (auction.bid(userId, itemId, bidAmount, authToken) == true){
        System.out.println("UserID: " + userId + " has placed a bid on itemID: "+ itemId+ " for: £" + bidAmount);
      }else {
        System.out.println("bid unsuccessful: token expired");
      }
    }catch (Exception e) {
      e.printStackTrace();
    } 
  }

  private static void sellItem(Auction auction, int userId, int auctionId) {
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

        //int auctionId = auction.newAuction(userId, item, authToken);
        System.out.println("New auction created with ID: " + auctionId);

        // Close auction
        System.out.print("Enter auction ID to close: ");
        auctionId = scanner.nextInt();
        scanner.nextLine(); // Consume newline

        AuctionResult result = auction.closeAuction(userId, auctionId, authToken);
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
            serverPublicKey = loadPublicKey("keys/server_public.key");
            KeyPair keyPair = generateKeyPair();
            clientPrivateKey = keyPair.getPrivate();
            clientPublicKey = keyPair.getPublic();


          //server connection
            String name = "Auction";
            Registry registry = LocateRegistry.getRegistry("localhost");
            Auction auction = (Auction) registry.lookup(name);
            userId = auction.register("user@example.com", clientPublicKey);
            authenticate(auction);
            int auctionid = createAuction(auction, userId);
            System.out.println("Registered with UserID: " + userId);            // implement some logical validation
            
            
            while (true) {
              //register user
              System.out.println("Select an option:");
              System.out.println("1. Sell an item");
              System.out.println("2. Buy an item");
              int choice = scanner.nextInt();
              scanner.nextLine();
              
              //createAuction(auction, userId);
              switch (choice) {
                case 1: // sell item
                  sellItem(auction, userId, auctionid);
                  break;
                case 2: //buy item
                  placeBid(auction, userId);
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