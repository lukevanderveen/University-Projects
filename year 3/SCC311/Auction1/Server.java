import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;
import javax.crypto.SealedObject;
import java.io.*;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;
import java.util.HashMap;

/*
 * commands:
 * rmiregistry
 * java Server
 * java Client
 * 
 * key generation happens here
 * encryption and decryption happens here
 *      encrypt sealedObject then send to client
 */

public class Server implements Auction {

    private HashMap<Integer, AuctionItem> auctionItems;

    private HashMap<Integer, String> users; // Store userID to email mapping
    private HashMap<Integer, AuctionSaleItem> ongoingAuctions; // Store ongoing auctions
    private int userIdCounter = 0; // Counter for unique user IDs
    private int auctionIdCounter = 0; // Counter for unique auction IDs


    public Server() throws RemoteException {
        super();

        auctionItems = new HashMap<>();
        users = new HashMap<>();
        ongoingAuctions = new HashMap<>();

        AuctionItem item;

        for (int i = 0; i < 10; i++) { 
            item = new AuctionItem();
            item.itemID = i;
            item.name = "Item" + i;
            item.description = "Description" + i;
            item.highestBid = 0;
            auctionItems.put(i, item);
            //System.out.println("Item" + i + " added to auctionItems: " + 
                //"ID: " + item.itemID + ", Name: " + item.name + 
                //", Description: " + item.description + 
                //", Highest Bid: " + item.highestBid);
        } 
     }
     
/* 
    public Auction getSpec(int itemID) throws RemoteException { 
        System.out.println("client request handled");
        AuctionItem item = auctionItems.get(itemID);
        try {
            Cipher cipher = Cipher.getInstance("AES");
            cipher.init(Cipher.ENCRYPT_MODE, secretKey);
            return new SealedObject(item, cipher);
        } catch (Exception e) {
            throw new RemoteException("Error encrypting item", e);
        }
    }*/
  
     public static void main(String[] args) {
        try {
         Server s = new Server();
         String name = "Auction";
         Auction stub = (Auction) UnicastRemoteObject.exportObject(s, 0);
         Registry registry = LocateRegistry.getRegistry();
         registry.rebind(name, stub);
         System.out.println("Server ready");
        } catch (RemoteException e) {
            System.out.println("RemoteException: ");
            e.printStackTrace();
        }catch (Exception e) {
         System.err.println("Exception:");
         e.printStackTrace();
        }
    }

    public boolean checkRegistered(int userID){
        return users.containsKey(userID);
    }

    public boolean checkItem(int itemID){
        return auctionItems.containsKey(itemID);
    }

    public int getHighestBid(AuctionItem item, int itemID){
        if (item == null){
            return -1;
        }else {
            return item.highestBid;
        }
        
    }

    public int register(String email) throws RemoteException {
        int userId = ++userIdCounter; // increment, assign user ID
        users.put(userId, email);
        return userId;
    }

    public int newAuction(int userID, AuctionSaleItem item) throws RemoteException {
        int auctionId = ++auctionIdCounter; // ncrement, assign auction ID
        ongoingAuctions.put(auctionId, item);
        return auctionId;
    }

    public AuctionItem[] listItems() throws RemoteException {
        return auctionItems.values().toArray(new AuctionItem[0]);
    }

    public boolean bid(int userID, int itemID, int price) throws RemoteException {
        AuctionItem item = auctionItems.get(itemID);
        if (checkRegistered(userID) == true){
            if(checkItem(itemID) == true){
               if (item != null && price > item.highestBid) {
                item.highestBid = price; // update highest bid
                return true;
               }  
            }
        }
        return false;   
    }

    public AuctionResult closeAuction(int userID, int itemID) throws RemoteException {
        AuctionItem item = auctionItems.get(itemID);
        if (item != null) {
            AuctionResult result = new AuctionResult();
            result.winningEmail = users.get(userID); 
            result.winningPrice = item.highestBid; 
            auctionItems.remove(itemID); 
            return result;
        }
        return null; // auction not found
    }

    public AuctionItem getSpec(int itemID) throws RemoteException {
        System.out.println("client request handled");
        AuctionItem item = auctionItems.get(itemID);
        return item;
    }
}


/*
 LEGACY CODE

    private SecretKey secretKey; // AES key
    private static final String KEY_DIRECTORY = "keys";
    private static final String KEY_FILE = "testKey.aes";

 private SecretKey generateKey() throws Exception {
        KeyGenerator KGen = KeyGenerator.getInstance("AES");
        KGen.init(128);
        return KGen.generateKey();
    }

    private void saveKey(SecretKey secretKey) throws Exception {
        byte[] data = secretKey.getEncoded(); 
        try (FileOutputStream f = new FileOutputStream(new File(KEY_DIRECTORY, KEY_FILE))) {
            f.write(data); 
        }
    }

    private SecretKey loadKey() throws Exception {
        File kFile = new File(KEY_DIRECTORY, KEY_FILE);
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

    SERVER INITALISATION: 
    // load or generate key
        try {
            secretKey = loadKey();
            if (secretKey == null) {
                secretKey = generateKey();
                saveKey(secretKey);
            }
        } catch (Exception e) {
            throw new RemoteException("Error loading or generating key", e);
        }
 */