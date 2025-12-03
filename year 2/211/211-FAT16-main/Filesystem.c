#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>


typedef struct __attribute__((__packed__)) {
    uint8_t BS_jmpBoot[ 3 ]; // x86 jump instr. to boot code
    uint8_t BS_OEMName[ 8 ]; // What created the filesystem
    uint16_t BPB_BytsPerSec; // Bytes per Sector                    *
    uint8_t BPB_SecPerClus; // Sectors per Cluster                  *
    uint16_t BPB_RsvdSecCnt; // Reserved Sector Count               *
    uint8_t BPB_NumFATs; // Number of copies of FAT                 *
    uint16_t BPB_RootEntCnt; // FAT12/FAT16: size of root DIR       *
    uint16_t BPB_TotSec16; // Sectors, may be 0, see below          *
    uint8_t BPB_Media; // Media type, e.g. fixed
    uint16_t BPB_FATSz16; // Sectors in FAT (FAT12 or FAT16)        *
    uint16_t BPB_SecPerTrk; // Sectors per Track
    uint16_t BPB_NumHeads; // Number of heads in disk
    uint32_t BPB_HiddSec; // Hidden Sector count
    uint32_t BPB_TotSec32; // Sectors if BPB_TotSec16 == 0          *
    uint8_t BS_DrvNum; // 0 = floppy, 0x80 = hard disk
    uint8_t BS_Reserved1; //
    uint8_t BS_BootSig; // Should = 0x29
    uint32_t BS_VolID; // 'Unique' ID for volume
    uint8_t BS_VolLab[ 11 ]; // Non zero terminated string          *
    uint8_t BS_FilSysType[ 8 ]; // e.g. 'FAT16 ' (Not 0 term.)
} BootSector;



typedef struct __attribute__((__packed__)){
    uint8_t DIR_Name[ 11 ]; // Non zero terminated string
    uint8_t DIR_Attr; // File attributes
    uint8_t DIR_NTRes; // Used by Windows NT, ignore
    uint8_t DIR_CrtTimeTenth; // Tenths of sec. 0...199
    uint16_t DIR_CrtTime; // Creation Time in 2s intervals
    uint16_t DIR_CrtDate; // Date file created
    uint16_t DIR_LstAccDate; // Date of last read or write
    uint16_t DIR_FstClusHI; // Top 16 bits file's 1st cluster
    uint16_t DIR_WrtTime; // Time of last write
    uint16_t DIR_WrtDate; // Date of last write
    uint16_t DIR_FstClusLO; // Lower 16 bits file's 1st cluster
    uint32_t DIR_FileSize; // File size in bytes
} DirectoryEntry;


int clusterFind(uint16_t fArray[], uint16_t cluster_num){
    if (cluster_num >= 0xfff8){
        printf("end of file\n");
        return 0;
    }else {
        printf("#%d\n", cluster_num);
        clusterFind(fArray, fArray[cluster_num]);
    }
    return 0;
}



int main(){
//task 1
    BootSector Boot;
    int f = open("fat16.img", O_RDONLY);
    read(f, &Boot, sizeof(BootSector));

    int Fat_size = (Boot.BPB_FATSz16*Boot.BPB_BytsPerSec)/2;
    uint16_t FAT[Fat_size];


//task 2
  
    printf("Bytes per sector: %d\n", Boot.BPB_BytsPerSec);
    printf("Sector per cluster: %d\n", Boot.BPB_SecPerClus);
    printf("Reversed sector count: %d\n", Boot.BPB_RsvdSecCnt);     
    printf("Copies of FAT: %d\n", Boot.BPB_NumFATs);
    printf("Size of root DIR: %d\n", Boot.BPB_RootEntCnt);
    printf("Total sectors in FAT: %d\n", Boot.BPB_TotSec16);
    printf("Total size of FAT: %d\n", Boot.BPB_FATSz16);
    printf("Sectors if BPB_TotSec16 == 0: %d\n", Boot.BPB_TotSec32);
    printf("Non zero terminated string: %s\n", Boot.BS_VolLab);
    printf("Fat_size: %d\n",Fat_size);
    printf("\n");


//task 3 

    //is reading FAT however is reading 62 bits ahead of where it should    
    lseek(f, 2048, SEEK_SET);
    read(f, FAT, Fat_size);

    clusterFind(FAT, 136);      //136 as based on the fat table's values this is the first cluster in the cluster chain

    
    /*          print out the fat table
    printf("\n");
    for (int i = 0; i < Fat_size; i++){
         printf("#%d : %d\n", i, FAT[i]);
    }
   */

//task 4    BPB_RsvdSecCnt + BPB_NumFATs * BPB_FATSz16. = first sector of rootDirectory

    uint16_t rootSize = Boot.BPB_RootEntCnt*32; //32=bytes per entry
    uint8_t Root[rootSize];
    struct DirectoryEntry* entries[Boot.BPB_RootEntCnt];

    lseek(f,  (Boot.BPB_RsvdSecCnt + Boot.BPB_NumFATs * Boot.BPB_FATSz16)*Boot.BPB_BytsPerSec, SEEK_SET);
    read(f, Root, rootSize);

    
    

    close(f);
}