#include <cstdlib>
#include <fstream>
#include <iostream>

using std::cerr;
using std::cout;
using std::endl;

int main(int argc, char *argv[]) {

    std::ofstream codif("codif.dat", std::ios_base::binary);

    unsigned short polai;
    unsigned short polaq;
    
    unsigned short polbi;
    unsigned short polbq;

    unsigned char sic[8];

    for (int ifpga = 0; ifpga < 48; ++ifpga) {

        for (int isamp = 0; isamp < 128; ++isamp) {

            for (int ichan = 0; ichan < 7; ++ichan) {

                polai = ((ifpga << 10) | (isamp << 2) | 0x0);
                polaq = ((ifpga << 10) | (isamp << 2) | 0x2);
                polbi = ((ifpga << 10) | (isamp << 2) | 0x1);
                polbq = ((ifpga << 10) | (isamp << 2) | 0x3);
             
                sic[0] = ((polbq & 0xff00) >> 8);
                sic[1] = (polbq & 0x00ff);
                sic[2] = ((polbi & 0xff00) >> 8);
                sic[3] = (polbi & 0x00ff);
                sic[4] = ((polaq & 0xff00) >> 8);
                sic[5] = (polaq & 0x00ff);
                sic[6] = ((polai & 0xff00) >> 8);
                sic[7] = (polai & 0x00ff);

                cout << (int)sic[0] << " " << (int)sic[1] << endl;
                
                codif.write(reinterpret_cast<char*>(sic), 8);

            }         
        }
    }

    if(!codif) {
        cerr << "Could not create the file" << endl;
        exit(EXIT_FAILURE);
    }

    codif.close();

    return 0;
}
