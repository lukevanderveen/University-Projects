import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;
import javax.crypto.SealedObject;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.attribute.UserDefinedFileAttributeView;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;
import java.security.KeyFactory;
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.Signature;
import java.security.spec.PKCS8EncodedKeySpec;
import java.security.spec.X509EncodedKeySpec;
import java.util.HashMap;
import java.util.UUID;

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
    private HashMap<Integer, String> users;
    private HashMap<Integer, AuctionSaleItem> ongoingAuctions; // Store ongoing auctions
    private HashMap<Integer, PublicKey> userPublicKeys; // Store userID to public key mapping
    private HashMap<Integer, String> userTokens;
    private HashMap<Integer, Long> tokenExpirations;
    private HashMap<Integer, String> userChallenges;
    private int userIdCounter = 0;
    private int auctionIdCounter = 0;

    private final PrivateKey serverPrivateKey;
    private final PublicKey serverPublicKey;


    public Server() throws RemoteException {
        super();

        auctionItems = new HashMap<>();
        users = new HashMap<>();
        ongoingAuctions = new HashMap<>();
        userPublicKeys = new HashMap<>();
        userTokens = new HashMap<>();
        tokenExpirations = new HashMap<>();
        userChallenges = new HashMap<>();

        try {
            generateAndSaveKeys("keys/server_private.key", "keys/server_public.key");
        } catch (NoSuchAlgorithmException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        serverPrivateKey = loadServerPrivateKey();
        serverPublicKey = loadServerPublicKey();

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
     
     private static void generateAndSaveKeys(String privateKeyPath, String publicKeyPath) throws NoSuchAlgorithmException, IOException {
        // Generate a 2048-bit RSA key pair
        KeyPairGenerator keyGen = KeyPairGenerator.getInstance("RSA");
        keyGen.initialize(2048);
        KeyPair pair = keyGen.generateKeyPair();

        // extract public private keys
        PrivateKey privateKey = pair.getPrivate();
        PublicKey publicKey = pair.getPublic();

        // save private key
        Files.write(Paths.get(privateKeyPath), privateKey.getEncoded());
        System.out.println("Private key saved to " + privateKeyPath);

        // save public key
        Files.write(Paths.get(publicKeyPath), publicKey.getEncoded());
        System.out.println("Public key saved to " + publicKeyPath);
    }
    

     private PrivateKey loadServerPrivateKey() {
        try {
            byte[] keyBytes = Files.readAllBytes(Paths.get("keys/server_private.key"));
            PKCS8EncodedKeySpec keySpec = new PKCS8EncodedKeySpec(keyBytes);
            KeyFactory keyFactory = KeyFactory.getInstance("RSA");
            return keyFactory.generatePrivate(keySpec);
        } catch (Exception e){
            try {
                throw new IOException("Error loading server private key", e);
            } catch (IOException e1) {
                e1.printStackTrace();
            }
        }
        return null;
    }
    
    private PublicKey loadServerPublicKey() {
        try{
            byte[] keyBytes = Files.readAllBytes(Paths.get("keys/server_public.key"));
            X509EncodedKeySpec keySpec = new X509EncodedKeySpec(keyBytes);
            KeyFactory keyFactory = KeyFactory.getInstance("RSA");
            return keyFactory.generatePublic(keySpec);
        }catch (Exception e){
            try {
                throw new IOException("Error loading server public key", e);
            } catch (IOException e1) {
                e1.printStackTrace();
            }
        }
        return null;
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

    public synchronized int register(String email, PublicKey pkey) throws RemoteException {
        int userId = ++userIdCounter;
        users.put(userId, email);
        userPublicKeys.put(userId, pkey);
        System.out.println("regestration: userid = "+userId+" associated key: "+userPublicKeys.get(userId));
        return userId;
    }

    public ChallengeInfo challenge(int userID, String clientChallenge) throws RemoteException {
        try {
            Signature sig = Signature.getInstance("SHA256withRSA");
            sig.initSign(serverPrivateKey);
            sig.update(clientChallenge.getBytes());
            byte[] sigChallenge = sig.sign();

            String serverChallenge = UUID.randomUUID().toString();
            userChallenges.put(userID, serverChallenge);

            ChallengeInfo chalInfo = new ChallengeInfo();
            chalInfo.response = sigChallenge;
            chalInfo.serverChallenge = serverChallenge;
            return chalInfo;

        }catch (Exception e){
            throw new RemoteException("Error generating challenge", e);
        }
    }

    public TokenInfo authenticate(int userID, byte[] signature) throws RemoteException {
        try{
            PublicKey clientpubKey = userPublicKeys.get(userID);
            if (clientpubKey == null){
                throw new RemoteException("User not registered");
            }   

            String serverChal = userChallenges.get(userID);
            if (serverChal == null) {
                throw new RemoteException("Challenge not found for user");
            }

            Signature sig = Signature.getInstance("SHA256withRSA");
            sig.initVerify(clientpubKey);
            sig.update(serverChal.getBytes());

            if (sig.verify(signature)) {
                String token = UUID.randomUUID().toString();
                long expiryTime = System.currentTimeMillis() + 10000;  

                userTokens.put(userID, token);
                tokenExpirations.put(userID, expiryTime);

                TokenInfo tokenI = new TokenInfo();
                tokenI.token = token;
                tokenI.expiryTime = expiryTime;

                userChallenges.remove(userID);
                return tokenI;
            }else{
                throw new RemoteException("Authentication failed: Invalid signature");
            }
        }catch (Exception e){
            System.out.println(e);
        }
        throw new RemoteException("Authentication failed: Invalid signature");
    }

    private boolean isTokenValid(int userID, String token) {
        System.out.println("Server 230: Validating token for userID: " + userID + ", token: " + token);
        String t = userTokens.get(userID); 
        Long expiry = tokenExpirations.get(userID); 

        if (t == null  || !token.equals(t)) {
            System.out.println("235: No token found for userID: " + userID+ " userTokens: "+ userTokens.get(userID)+"/ token doesn't match");
            return false;
        }
        if (expiry == null) { 
            System.out.println("No expiration found for userID: " + userID);
            return false;
        }
        return token.equals(t) && System.currentTimeMillis() < expiry;
    }

    public synchronized int newAuction(int userID, AuctionSaleItem item, String token) throws RemoteException {
        if (userID <= 0 || token == null || item == null) {
            System.out.println("server 246: userID = "+userID +" token: "+token+ " item: " + item.name);
            throw new RemoteException("Invalid parameters provided for newAuction.");
        }
        if (!isTokenValid(userID, token)) {
            throw new RemoteException("Authentication failed or token expired.");
        }
        int auctionId = ++auctionIdCounter;
        ongoingAuctions.put(auctionId, item);  // create new AuctionItem based on AuctionSaleItem
        return auctionId;
    }

    public AuctionItem[] listItems(int userID, String token) throws RemoteException {
        if (!isTokenValid(userID, token)) {
            return null;
        }
        return auctionItems.values().toArray(new AuctionItem[0]);
    }

    public synchronized boolean bid(int userID, int itemID, int price, String token) throws RemoteException {
        if (!isTokenValid(userID, token)) {
            return false;
        }
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

    public synchronized AuctionResult closeAuction(int userID, int itemID, String token) throws RemoteException {
        if (!isTokenValid(userID, token)) {
            return null;
        }
        AuctionItem item = auctionItems.get(itemID);
        if (item != null) {
            AuctionResult result = new AuctionResult();
            result.winningEmail = users.get(userID);
            result.winningPrice = item.highestBid;
            auctionItems.remove(itemID); 
            return result;
        }
        return null;
    }

    public AuctionItem getSpec(int userID, int itemID, String token) throws RemoteException {
        if (!isTokenValid(userID, token)) {
            return null;
        }
        return auctionItems.get(itemID);
    }

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