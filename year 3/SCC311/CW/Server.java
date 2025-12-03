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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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

    private SecretKey secretKey; // AES key
    private static final String KEY_DIRECTORY = "keys";
    private static final String KEY_FILE = "testKey.aes";

    private HashMap<Integer, List<Integer>> userAuctionItems; // userId -> List of itemIDs
    private HashMap<Integer, HashMap<Integer, AuctionItem>> auctionItems; //auctionId -> (itemId -> item)
    private HashMap<Integer, String> users; // store userID to email 
    private HashMap<Integer, AuctionSaleItem> ongoingAuctions; // Store auctions
    private HashMap<Integer, HashMap<Integer, AuctionSaleItem>> auctionItemSellers; // auctionId -> (itemId -> userId)
    private int userIdCounter = 0; // Counter for unique user IDs
    private int auctionIdCounter = 0; // Counter for unique auction IDs

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

    public Server() throws RemoteException {
        super();

        userAuctionItems = new HashMap<>();
        users = new HashMap<>();
        ongoingAuctions = new HashMap<>();
        auctionItemSellers = new HashMap<>();
        auctionItems = new HashMap<>();

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
    

        AuctionItem item;

        for (int i = 0; i < 10; i++) { 
            item = new AuctionItem();
            item.itemID = i;
            item.name = "Item" + i;
            item.description = "Description" + i;
            item.highestBid = 0;
            //auctionItems.put(0, item);
            auctionItems.putIfAbsent(0, new HashMap<>());
            auctionItems.get(0).put(item.itemID, item);;
            
            userAuctionItems.putIfAbsent(0, new ArrayList<>());
            userAuctionItems.get(0).add(item.itemID);;

            //auctionItemSellers.putIfAbsent(0, new HashMap<>());
            //auctionItemSellers.get(0).put(item.itemID, item);;
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

    public boolean checkItem(int itemID, int auctionID){
        if (auctionItems.containsKey(auctionID)){
            HashMap<Integer, AuctionItem> innerMap = auctionItems.get(auctionID);
            if (innerMap.containsKey(itemID)){
                return innerMap.containsKey(itemID);
            }
            return false;
        }
        return false;
    }

    public int getHighestBid(AuctionItem item, int itemID){
        if (item == null){
            return -1;
        }else {
            return item.highestBid;
        }
    }

    public AuctionItem getItem(int itemID) {
        for (Map.Entry<Integer, HashMap<Integer, AuctionItem>> entry : auctionItems.entrySet()){
            int auctionID = entry.getKey();
            HashMap<Integer, AuctionItem> innerMap = entry.getValue();
            if(checkItem(itemID , auctionID) == true){
                AuctionItem item = innerMap.get(itemID);
                return item;
            }
        }
        return null;
    }

    public int register(String email) throws RemoteException {
        int userId = ++userIdCounter; // increment, assign user ID
        users.put(userId, email);
        System.out.println("user regestered ID = "+ userId);
        return userId;
    }

    public int newAuction(int userID, AuctionSaleItem item) throws RemoteException {
        int auctionId = ++auctionIdCounter; // ncrement, assign auction ID
        ongoingAuctions.put(auctionId, item);

        AuctionItem auctionItem = new AuctionItem(); // Assuming you have a constructor for AuctionItem
        auctionItem.itemID = auctionId; // Set the itemID
        auctionItem.name = item.name; // Assuming AuctionSaleItem has these fields
        auctionItem.description = item.description;
        auctionItem.highestBid = 0;

        auctionItems.putIfAbsent(auctionId, new HashMap<>());
        auctionItems.get(auctionId).put(auctionItem.itemID, auctionItem);

        auctionItemSellers.putIfAbsent(auctionId, new HashMap<>());
        auctionItemSellers.get(auctionId).put(userID, item);
        return auctionId;
    }

    public AuctionItem[] listItems() throws RemoteException {
        return auctionItems.values().toArray(new AuctionItem[0]);
    }

    public boolean bid(int userID, int itemID, int price) throws RemoteException {
        if (checkRegistered(userID) == true){
            for (Map.Entry<Integer, HashMap<Integer, AuctionItem>> entry : auctionItems.entrySet()){
                int auctionID = entry.getKey();
                HashMap<Integer, AuctionItem> innerMap = entry.getValue();
                if(checkItem(itemID , auctionID) == true){
                    AuctionItem item = innerMap.get(itemID);
                    if (item != null && price > item.highestBid) {
                        item.highestBid = price; // update highest bid
                        return true;
                    }  
                }
            }
        }
        return false;  
    }

    public AuctionResult closeAuction(int userID, int itemID) throws RemoteException {

        System.out.println("Attempting to close auction for itemID: " + itemID + " by userID: " + userID);
        AuctionItem item = getItem(itemID);
        System.out.println("Auction item retrieved: " + (item != null ? "Exists" : "Does not exist"));

        if (item != null) {
            if (checkRegistered(userID) == true){// redundent due to check later on
                System.out.println("User registration check for userID: " + userID);

                try {
                    System.out.println("User registration check for userID: " + userID);
                    System.out.println("userAuctionItems containsKey(userID): " + userAuctionItems.containsKey(userID));

                    if (userAuctionItems.containsKey(userID)){

                        List<Integer> userItems = userAuctionItems.get(userID);
                        System.out.println("userAuctionItems.get(userID): " + (userItems != null ? userItems.toString() : "null"));

                        if (userItems != null && userItems.contains(itemID)){

                            System.out.println("userID " + userID + " is selling itemID " + itemID);
                            AuctionResult result = new AuctionResult();
                            result.winningEmail = users.get(userID); 
                            result.winningPrice = item.highestBid; 
                            System.out.println("Removing itemID " + itemID + " from auctionItems");
                            auctionItems.remove(itemID); 
                            System.out.println("Auction successfully closed for itemID: " + itemID);
                            return result;

                        }else {
                            System.out.println("User " + userID + " is not the seller of itemID " + itemID);
                        }
                    }else {
                        System.out.println("userAuctionItems does not contain userID: " + userID);
                    }

                } catch (Exception e) {
                    System.out.println("Exception occurred during auction closure: " + e.getMessage());
                    e.printStackTrace();
                    System.out.println("User is not the seller of this item.");
                }   
            }else {
                System.out.println("User " + userID + " is not registered.");
            }
        }else{
            System.out.println("Auction item with itemID " + itemID + " not found in auctionItems.");
        }
        System.out.println("Failure to close auction for itemID: " + itemID);
        return null; // auction not found
    }


    public AuctionItem getSpec(int itemID) throws RemoteException {
        System.out.println("client request handled");
        AuctionItem item = getItem(itemID);
        return item;
    }
}


/*
    LEGACY CODE:
* FROM CLOSE AUCTION INNNER SELECTION

&&  && userAuctionItems.get(userID).contains(itemID)) {
    System.out.println("userID is selling itemID"+ itemID +": " + userAuctionItems.get(userID).contains(itemID));//here: NullPointerException
    System.out.println("");

 */