import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;
import java.util.ArrayList;
import java.util.List;

public class FrontEnd implements Auction {

    private List<Integer> replicaIDs = new ArrayList<>(); // IDs of replicas
    private int primaryReplicaID = -1;    // Current primary replica
    private String serviceName = "Replica";

    public FrontEnd() throws RemoteException {
        updateReplicaIDs();
        selectPrimary();
    }

    private void updateReplicaIDs() {
        try {
            Registry registry = LocateRegistry.getRegistry("localhost");
            replicaIDs.clear(); // Clear existing list
            for (String name : registry.list()) {
                if (name.startsWith(serviceName)) {
                    int id = Integer.parseInt(name.replace(serviceName, ""));
                    if (!replicaIDs.contains(id)) {
                        replicaIDs.add(id);
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Failed to update replica IDs: " + e.getMessage());
        }
    }

    

    private void selectPrimary() {
        primaryReplicaID = -1; // clear current primary
        for (int replicaID : replicaIDs){
            try {
                Auction rep = getReplicaStub(replicaID);
                int primaryID = rep.getPrimaryReplicaID();
                if (primaryID != -1){
                    this.primaryReplicaID = primaryID;
                    System.out.println("primary replica selected: "+ primaryReplicaID); 
                    notifyBackups(primaryID);
                    return;
                }
            } catch (Exception e){
                System.out.println("couldn't contact replica : " + replicaID + ": "+ e.getMessage());
            }
        }
        System.out.println("no primary replica available, waiting for election...");
    }

    private Auction triggerElection() {  // trigger an election
        System.out.println("Initiating election...");
        Auction newPrimary = null;
        List<Integer> failedReplicas = new ArrayList<>();
        for (int replicaID : replicaIDs) {
            if (failedReplicas.contains(replicaID)) continue;
            try {
                Auction replica = getReplicaStub(replicaID);
                AuctionItem[] items = replica.listItems(); // trigger an syncronisation (i've adjusted a few of my auction methods to be dual perpose)
                selectPrimary();
                if (primaryReplicaID != -1) {
                    newPrimary = getReplicaStub(primaryReplicaID);
                    return newPrimary;
                }
            } catch (Exception e) {
                System.err.println("Failed to contact replica " + replicaID + ": " + e.getMessage());
                failedReplicas.add(replicaID);
            }
        }
        selectPrimary();
        try{
            return primaryReplicaID != -1 ? getReplicaStub(primaryReplicaID) : null;
        }catch (Exception e){
            System.err.println("Failed to get replica stub " + primaryReplicaID + ": " + e.getMessage());
            return null;
        }
    }

    private void notifyBackups(int primaryID) { // notify other backups of new primary and syncronisations
        try {
            Registry reg = LocateRegistry.getRegistry("localhost");
            for (String name : reg.list()) {
                if (name.startsWith(serviceName)) {
                    int id = Integer.parseInt(name.replace(serviceName, ""));
                    if (id != primaryID) { // Notify all except the new primary
                        Auction backup = (Auction) reg.lookup(name);
                        try {
                            backup.listItems();
                            System.out.println("Triggered synchronization with Replica " + id);
                        } catch (Exception e) { // help narrow down issue
                            System.err.println("Failed to synchronize Replica " + id + ": " + e.getMessage());
                        }
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Failed to notify backups: " + e.getMessage());
        }
    }

    private Auction getReplicaStub(int replicaID) { // get a replica stub
        try{
            Registry registry = LocateRegistry.getRegistry("localhost");
            return (Auction) registry.lookup(serviceName + replicaID);
        }catch (Exception e){
            System.err.println("Failed to get stub for Replica " + replicaID + ": " + e.getMessage());
            return null;
        }
    }

    // forward requests to primary
    private <T> T forward(RequestHandler<T> handler) throws RemoteException{
        if (primaryReplicaID == -1){ // if primary id == -1 (there isn't a primary)
            System.out.println("No primary available. Triggering election...");
            Auction newPrimary = triggerElection(); // trigger an election
            if (newPrimary == null)throw new RemoteException("no primary available");
        }

        try { // get primary and send handler request ot primary replica
            Auction primary = getReplicaStub(primaryReplicaID);
            return handler.handle(primary);
        } catch (Exception e){ // exception
            System.err.println("primary replica failed: "+ e.getMessage());
            System.out.println("Attempting election...");
            Auction newPrimary = triggerElection(); // trigger an election
            if (newPrimary == null)throw new RemoteException("no primary available");
            try {
                return handler.handle(newPrimary);
            } catch (Exception e1) {
                e1.printStackTrace();
                return null;
            }
        }
    }

    // =============Auction interface implementations==================
    

    public int register(String email) throws RemoteException {
        return forward(replica -> replica.register(email));
    }

    public int newAuction(int userID, AuctionSaleItem item) throws RemoteException {
        return forward(replica -> replica.newAuction(userID, item));
    }

    public AuctionItem[] listItems() throws RemoteException {
        return forward(Auction::listItems);
    }

    public boolean bid(int userID, int itemID, int price) throws RemoteException {
        return forward(replica -> replica.bid(userID, itemID, price));
    }

    public AuctionResult closeAuction(int userID, int itemID) throws RemoteException { // piggy back this function to ensure primary ID is changed
        if (userID == -1 && itemID > 0) { // Use special signal to update primaryReplicaID
            this.primaryReplicaID = itemID; // Update with the new primary ID
            System.out.println("FrontEnd updated with new primary replica ID: " + primaryReplicaID);
            return null; // No actual auction result
        }
        return forward(replica -> replica.closeAuction(userID, itemID));
    }

    public AuctionItem getSpec(int itemID) throws RemoteException {
        return forward(replica -> replica.getSpec(itemID));
    }

    public int getPrimaryReplicaID() throws RemoteException {
        return primaryReplicaID;
    }

    private interface RequestHandler<T> {
        T handle(Auction replica) throws Exception;
    }

    public static void main (String[] args){
        try {
            // define replicas <- presents an issue 
            FrontEnd fe = new FrontEnd();
            Auction stub = (Auction) UnicastRemoteObject.exportObject(fe, 0);
            Registry reg = LocateRegistry.getRegistry();
            reg.rebind("FrontEnd", stub);

            System.out.println("FrontEnd is running and connected to primary replica: " + fe.getPrimaryReplicaID());
        } catch (Exception e){
            System.err.println("frontend error: "+ e.getMessage());
        }
    } 

}


/*
 * 

public Auction triggerElection(){
        System.out.println("No primary replica available, initiating election...");
        Auction newPrimary = null;
        List<Integer> failedReplicas = new ArrayList<>();

        try {
            Registry reg = LocateRegistry.getRegistry("localhost");
            
            for (String name : reg.list()){
                if (name.startsWith("Replica")){
                    int replicaID = Integer.parseInt(name.replace("Replica", ""));
                    if (failedReplicas.contains(replicaID)) continue; // dynamcically mark failed replicas

                    try {
                        Auction replica = (Auction) reg.lookup(name);
                        AuctionItem[] items = replica.listItems(); // listItems, modified so that it calls election() if there is no primaryID  

                        if (primaryReplicaID != -1) {
                            System.out.println("Primary identified during listItems call: " + primaryReplicaID);
                            newPrimary = getReplicaStub(primaryReplicaID);
                            return newPrimary; // Return the new primary stub
                        }
                    } catch (Exception e){
                        System.err.println("Failed to contact replica " + replicaID + ": " + e.getMessage());
                        failedReplicas.add(replicaID); // mark replica 
                    }
                    
                }
            }
            selectPrimary();
            if (primaryReplicaID != -1){
                newPrimary = getReplicaStub(primaryReplicaID);
            }
        } catch(Exception e){
            System.err.println("Election failed: " + e.getMessage());
        }
        if (primaryReplicaID == -1) {
            System.err.println("Election could not establish a primary replica.");
        }
        return newPrimary;
    }

    // forward requests to primary
    private <T> T forward(RequestHandler<T> handler) throws RemoteException{
        if (primaryReplicaID == -1){ // if primary id == -1 (there isn't a primary)
            System.out.println("No primary available. Triggering election...");
            Auction newPrimary = triggerElection(); // trigger an election

            if (newPrimary == null)throw new RemoteException("no primary available");
            try {
                return handler.handle(newPrimary);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        try { // get primary and send handler request ot primary replica
            Auction primary = getReplicaStub(primaryReplicaID);
            return handler.handle(primary);
        } catch (Exception e){ // exception
            System.err.println("primary replica failed: "+ e.getMessage());
            System.out.println("Attempting election...");
            Auction newPrimary = triggerElection(); // trigger an election
            if (newPrimary == null)throw new RemoteException("no primary available");
            try {
                return handler.handle(newPrimary);
            } catch (Exception e1) {
                e1.printStackTrace();
            }
        }
        return null;
    }

 */