import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/*
* commands:
* rmiregistry
* java Replica n 
* java FrontEnd
* java Client

*/

public class Replica implements Auction {

    private int replicaID;
    private boolean primary = false;

    private HashMap<Integer, AuctionItem> auctionItems = new HashMap<>();;
    private HashMap<Integer, String> users = new HashMap<>();; // Store userID to email mapping
    private HashMap<Integer, AuctionSaleItem> ongoingAuctions = new HashMap<>();; // Store ongoing auctions
    private int userIdCounter = 0; // Counter for unique user IDs
    private int auctionIdCounter = 0; // Counter for unique auction IDs


    public Replica(int id) throws RemoteException {
        super();
        this.replicaID = id;
        this.primary = (id == 1);
        System.out.println("Replica " + id + " is " + (primary ? "PRIMARY" : "BACKUP")); //if it is primary replica (id 1 initally) PRIMARY, if not BACKUP
        System.out.println("primary ID = " + getPrimaryReplicaID());
        AuctionItem item;

        if (primary){
            for (int i = 0; i < 10; i++) { 
                item = new AuctionItem();
                item.itemID = i;
                item.name = "Item" + i;
                item.description = "Description" + i;
                item.highestBid = 0;
                auctionItems.put(i, item);
            } 
        }else {
            synchronizeWithPrimary();
        }
    }

    public void notifyFrontEnd(){ // notify front end of new primary 
        try {
            Registry reg = LocateRegistry.getRegistry("localhost");
            Auction frontEndStub = (Auction) reg.lookup("FrontEnd");
            
            System.out.println("Notifying FrontEnd about new primary: " + this.replicaID);
            frontEndStub.closeAuction(-1, this.replicaID);  // notify front end of new primary dual purpose
            
            System.out.println("FrontEnd notified about new primary.");
        }catch (Exception e){
            System.err.println("Failed to notify components: " + e.getMessage());
        }
    }

    public void election(){
        System.out.println("Replica " + replicaID + " initiating election...");
        try {
            int highestID = -1;
            Registry reg = LocateRegistry.getRegistry("localhost");

            int currentPrimary = getPrimaryReplicaID(); // Use a method that checks current primary
            if (currentPrimary != -1) {
                System.out.println("Replica " + replicaID + " found current PRIMARY: " + currentPrimary);
                return; // No need to initiate election, primary already set
            }

            for (String name : reg.list()){
                if (name.startsWith("Replica")) {
                    int otherID = Integer.parseInt(name.replace("Replica", ""));
                    if (otherID > highestID) {
                        highestID = otherID;
                    }
                    if (highestID == replicaID){
                        break;
                    }
                }
            }
            // election
            if (highestID == replicaID){ // elect self it is the highest
                setPrimary(true);
                notifyFrontEnd();
                synchronizeWithBackups(); // sync with backups when electing/re-electing
                System.out.println("Replica " + replicaID + " is now PRIMARY.");
            } else {
                setPrimary(false);
                System.out.println("Replica " + replicaID + " is BACKUP. Current PRIMARY: " + highestID);
            }

            // notify backups of new election
            for (String name : reg.list()){
                if (name.startsWith("Replica")) {
                    int otherID = Integer.parseInt(name.replace("Replica", ""));
                    if (otherID != this.replicaID){
                        try {
                            Auction backupStub = (Auction) reg.lookup(name);
                            backupStub.closeAuction(1, highestID == otherID ? 0 : -1); //piggy back
                        }catch(Exception e){
                            System.err.println("Failed to notify Replica " + e.getMessage());
                        }
                    }
                }
            }
        }catch (Exception e){
            System.err.println("Election error: " + e.getMessage());
        }
    }

    public void setPrimary(boolean primary){
        this.primary = primary;
        System.out.println("Replica " + replicaID + " is now " + (primary ? "PRIMARY" : "BACKUP"));
    }

    public List<Integer> getOtherReplicaIDs() throws RemoteException {
        List<Integer> otherReplicas = new ArrayList<>();
        try {
            Registry registry = LocateRegistry.getRegistry("localhost");
            for (String name : registry.list()) {
                if (name.startsWith("Replica")){
                    int replicaID = Integer.parseInt(name.replace("Replica", ""));
                    if (replicaID != this.replicaID){ // get all other replicas (the one doing this process will be primary replica)
                        try {
                            Auction replicaStub = (Auction) registry.lookup(name);
                            replicaStub.getPrimaryReplicaID(); // Test connection
                            otherReplicas.add(replicaID);
                        } catch (Exception e) {
                            System.err.println("Replica " + replicaID + " is unreachable.");
                        }
                    }
                }
            }
        }catch (Exception e){
            System.err.println("Error discovering other replicas: " + e.getMessage());
        }
        return otherReplicas;
    }
    

    // syncronisation methods to allow backups to sync with the primary's request handling in order to maintain up to date knowledge of the auction's state

    private void synchronizeWithPrimary() { //perfomed by backup to pull the primary state in order to syncronise
        if (primary) return; // if primary won't need to syncronise with itself
        System.out.println("Replica " + replicaID + " (BACKUP) synchronizing state with PRIMARY...");

        try {
            int primaryID = getPrimaryReplicaID(); // Use getPrimaryReplicaID here
            if (primaryID != -1) {
                Registry registry = LocateRegistry.getRegistry("localhost");
                Auction primaryStub = (Auction) registry.lookup("Replica" + primaryID);
                AuctionItem[] items = primaryStub.listItems();
                auctionItems.clear();
                for (AuctionItem item : items) {
                    auctionItems.put(item.itemID, item);
                }
                System.out.println("Replica " + replicaID + " synchronized with PRIMARY.");
            } else {
                System.out.println("No primary found. Initiating election...");
                election();
            }
        } catch (Exception e) {
            System.err.println("Failed to synchronize with PRIMARY: " + e.getMessage());
            election();
        }
    }

    private void synchronizeWithBackups() throws RemoteException { // used by the primary to push the changes it's made to the other backups
        if (!primary) return; // ensures only primary can perform this
        System.out.println("Replica " + replicaID + " (PRIMARY) synchronizing changes with BACKUPS...");
        try{
            List<Integer> otherReplicas = getOtherReplicaIDs();
            Registry registry = LocateRegistry.getRegistry("localhost");
            for (int backUpID : otherReplicas){
                boolean success = false;
                int attempts = 0;
                while (!success && attempts < 3){
                    try {
                        Auction backUpStub = (Auction) registry.lookup("Replica" + backUpID);

                        System.out.println("Synchronizing with Backup Replica " + backUpID);
                        backUpStub.register(null);
                        success = true;
                    }catch (Exception e){
                        attempts++;
                        System.err.println("Failed to synchronize with Replica " + backUpID + ": " + e.getMessage());
                    }
                }
            }
        }catch (Exception e){
            System.err.println("Error during backup synchronization: " + e.getMessage());
        }
    }  
    
    public void updateState(AuctionItem[] updatedItems) throws RemoteException {
        if (primary) {
            System.err.println("Primary should not receive state updates.");
            return;
        }
        auctionItems.clear();
        for (AuctionItem item : updatedItems) {
            auctionItems.put(item.itemID, item);
        }
        System.out.println("Replica " + replicaID + " updated with new state.");
    }


    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.println("Usage: java Replica <replicaID>");
            return;
        }
        try {
            int id = Integer.parseInt(args[0]);
            Replica replica = new Replica(id);
            String name = "Replica" + id;
            Auction stub = (Auction) UnicastRemoteObject.exportObject(replica, 0);
            Registry registry = LocateRegistry.getRegistry();
            registry.rebind(name, stub);

            System.out.println("Replica " + id + " is ready");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    // helper functions
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
    
 // ==============================auction inteface implementations===================================
 //(a few of these have become dual purpose for primary election and syncronisation)

    public int register(String email) throws RemoteException {
        if (email == null) { // Special signal for state synchronization
            System.out.println("Replica " + replicaID + " (BACKUP) received state synchronization request.");
            updateState(auctionItems.values().toArray(new AuctionItem[0])); // Trigger updateState
            return -1; // No actual item is returned
        }
        System.out.println("Replica " + replicaID + " (PRIMARY) handling register request for email: " + email);
        if (!primary) throw new RemoteException("Not the primary replica");
        int userId = ++userIdCounter; // increment, assign user ID
        users.put(userId, email);
        synchronizeWithBackups();
        return userId;
    }

    public int newAuction(int userID, AuctionSaleItem item) throws RemoteException {
        System.out.println("Replica " + replicaID + " (PRIMARY) creating new auction for userID: " + userID + " with item: " + item.name);
        if (!primary) throw new RemoteException("Not the primary replica");
        int auctionId = ++auctionIdCounter; // ncrement, assign auction ID
        ongoingAuctions.put(auctionId, item);
        synchronizeWithBackups();
        return auctionId;
    }

    public AuctionItem[] listItems() throws RemoteException { 
        if (!primary) {
            System.out.println("Replica " + replicaID + " received synchronization request from PRIMARY.");
            synchronizeWithPrimary();  // Call the sync method to update backup's state
        }
        if (getPrimaryReplicaID() == -1){
            System.out.println("No primary detected. Initiating election..."); // piggy back off the auction methods to get around type casting issues between replica methods and auction methods
            election();
        }else {
            System.out.println("Primary exists. Synchronizing with primary...");
            synchronizeWithPrimary();
        }
        System.out.println("Replica " + replicaID + " listing items. Current role: " + (primary ? "PRIMARY" : "BACKUP"));
        return auctionItems.values().toArray(new AuctionItem[0]);
    }

    // debug function: check whether listItems post syncronisation actually syncronises

    public boolean bid(int userID, int itemID, int price) throws RemoteException {
        if (!primary) {
            System.out.println("Replica " + replicaID + " received a bid but is not PRIMARY.");
            int primaryID = getPrimaryReplicaID();
            if (primaryID == -1) {
                System.out.println("No primary detected. Initiating election...");
                election();
                throw new RemoteException("Election in progress. Try again.");
            } else {
                throw new RemoteException("Not the primary. Redirect to PRIMARY: " + primaryID);
            }
        }
        //logIfNotPrimary();
        System.out.println("Replica " + replicaID + " (PRIMARY) processing bid for userID: " + userID + ", itemID: " + itemID + ", bid: £" + price);
        if (!primary) throw new RemoteException("Not the primary replica");
        AuctionItem item = auctionItems.get(itemID);
        if (checkRegistered(userID) == true){
            if(checkItem(itemID) == true){
            if (item != null && price > item.highestBid) {
                item.highestBid = price; // update highest bid
                synchronizeWithBackups();
                return true;
            }  
            }
        }
        return false;   
    }

    public AuctionResult closeAuction(int userID, int itemID) throws RemoteException {
        if (userID == -1 && itemID > 0) { // Signal for primary election
            System.out.println("Replica " + replicaID + " received primary election signal.");
            setPrimary(true); // Promote to primary
            return null; // No actual auction result
        } 
        System.out.println("Replica " + replicaID + " (PRIMARY) closing auction for itemID: " + itemID + " by userID: " + userID);
        if (!primary) throw new RemoteException("Not the primary replica");
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
        System.out.println("Replica " + replicaID + " providing item specification for itemID: " + itemID);
        AuctionItem item = auctionItems.get(itemID);
        return item;
    }

    public int getPrimaryReplicaID() throws RemoteException{
        if (primary) return replicaID;
        try {
            Registry reg = LocateRegistry.getRegistry("localhost");
            for (String name : reg.list()){ // search through the replicas
                if (name.startsWith("Replica")){
                    Auction repStub = (Auction) reg.lookup(name);
                    int primaryID = repStub.getPrimaryReplicaID(); // recursively look through each replica to find primary ID
                    if (primaryID != -1) return primaryID;
                }
            }
        }catch (Exception e){
            System.err.println("Error while finding primary replica: " + e.getMessage());
        }
        return -1; //no primary
    }

}
/*
LEGACY CODE

*/