public class InventoryMain {

    public static void main (String[] args){
        int add = Integer.parseInt(args[0]);
        int remove = Integer.parseInt(args[1]);
        int tick = add + remove;
        InventoryThread[] threads = new InventoryThread[add + remove];
        Inventory I = new Inventory();



        //add Add threads to the threads array
        for (int i = 0; i < add; i++){
            threads[i] = new InventoryThread(1, I);
        }
        //add Remove threads to the threads array
        for (int j = 0; j < remove; j++){
            threads[add + j] = new InventoryThread(0, I);
        }
        //start and join the threads
        InventoryThread current;
        for (int r = 0; r < tick; r++){
            threads[r].start();
            current = threads[r];
            try {
                current.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        System.out.println("Final inventory size = " + I.getTotal());
    }
}
