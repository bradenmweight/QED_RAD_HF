import numpy as np
from time import time
from numba import njit

from pyscf import gto


def get_Globals():
    global L, NPW
    L      = 4.0
    NPW    = 11 # Number of plane-waves per dimension -- odd

    # Molecular System
    global coords, charges, n_elec_alpha, n_elec_beta
    mol          = gto.Mole()
    #mol.basis    = basis_set
    mol.unit     = 'Bohr'
    mol.symmetry = False
    mol.atom = 'He 0 0 0; He 0 0 1.5'
    mol.build()
    coords  = mol.atom_coords()
    charges = mol.atom_charges()
    n_elec_alpha, n_elec_beta = mol.nelec

def get_kGRID():
    kGRID = np.zeros( (NPW,NPW,NPW,3) ) # kx,ky,kz
    kVALS = np.arange( -NPW//2 + 1,NPW//2 + 1 ) * 2 * np.pi / L
    for nx , kx in enumerate(kVALS):
        for ny, ky  in enumerate(kVALS):
            for nz, kz in enumerate(kVALS):
                kGRID[nx,ny,nz,:] = np.array([kx,ky,kz]) # nx, ny, nz
    return kGRID.reshape( (NPW**3,3) )

@njit
def get_V_eN( k1, k2 ):
    kmk = k2 - k1
    if ( np.dot( kmk, kmk ) != 0.0 ):
        V_eN   =  4 * np.pi / np.dot( kmk, kmk )
        Rk     = np.sum( coords[:,:] * kmk[None,:], axis=-1 ) # "Rd,d->R"
        V_eN  *= np.sum( charges[:] * np.exp(1j * Rk ) ) # "R->"
        return V_eN
    else:
        return 0.0 + 0.0j

@njit
def get_T_e( k1 ):
    return 0.500 * np.dot( k1, k1 )

@njit
def get_1RDM( C, k3i, k4i ):
    return np.sum( C[k3i,:].conj() * C[k4i,:] )

@njit
def get_V_ee( k1i, k2i, k1, k2, C, kGRID ):
    """
    Chemists' Notation:
    J = [ 12 | 34 ] D_{34}
    K = [ 13 | 24 ] D_{34}
    Condition: k3 - k4 = k2 - k1 --> k3 = k2 - k1 + k4
    """
    kGRID = kGRID.reshape( (NPW,NPW,NPW,3) )
    V_ee_J = 0.0 + 0.0j
    V_ee_K = 0.0 + 0.0j

    ### Coulomb Term ###
    k1i  = np.array([k1i // NPW**2, k1i // NPW % NPW, k1i % NPW])
    k2i  = np.array([k2i // NPW**2, k2i // NPW % NPW, k2i % NPW])
    k21  = k2 - k1
    k34i = k2i - k1i

    '''
    if k34i < 0:
        for k3i from 0 to NPW + k34i
            k4i = k3i - k34i
    if k34i > 0:
        for k4i from 0 to NPW - k34i
            k3i = k4i + k34i
        for k3i from k34i to NPW
            k4i = k3i - k34i

    for k3i from k34i * (sign(k34i) + 1)/2 to NPW + k34i * (1 - sign(k34i))/2
        k4i = k3i - k34i
    '''



    if ( (k34i <= -NPW).any() or (k34i >= NPW).any() ):
        pass
    else:
        FACTOR_COUL =  4 * np.pi / np.dot( k2 - k1, k2 - k1 )
        for nx in range( k34i[0] * (np.sign(k34i[0]) + 1)//2 , NPW + k34i[0] * (1 - np.sign(k34i[0]))//2):
            for ny in range( k34i[1] * (np.sign(k34i[1]) + 1)//2 , NPW + k34i[1] * (1 - np.sign(k34i[1]))//2 ):
                for nz in range(k34i[2] * (np.sign(k34i[2]) + 1)//2 , NPW + k34i[2] * (1 - np.sign(k34i[2]))//2 ):
                    k3i = np.array([nx,ny,nz])
                    k4i = k3i - k34i
                    D34 = get_1RDM( C, k3i[0] * NPW**2 + k3i[1] * NPW + k3i[2], k4i[0] * NPW**2 + k4i[1] * NPW + k4i[2] )
                    # if ( np.sign(k34i[0]) * np.sign(k34i[1]) * np.sign(k34i[2]) < 0  ):
                    #     D34 = np.conj(D34)
                    V_ee_J += FACTOR_COUL * D34

    # ### Exchange Term ###
    k34i = k2i + k1i
    if ( (k34i >= 2*NPW).any()):
        pass
    else:
        for nx in range( (2*NPW - k34i[0]) * (k34i[0] > NPW) , NPW * (k34i[0] >= NPW) + (1 + k34i[0]) * (NPW > k34i[0])):
            for ny in range(  (2*NPW - k34i[1]) * (k34i[1] > NPW) , NPW * (k34i[1] >= NPW) + (1 + k34i[1]) * (NPW > k34i[1]) ):
                for nz in range( (2*NPW - k34i[2]) * (k34i[2] > NPW) , NPW * (k34i[2] >= NPW) + (1 + k34i[2]) * (NPW > k34i[2]) ):
                    
                    k3i = np.array([nx,ny,nz])
                    k4i = k34i - k3i
                    if ( (k4i < 0).any() or (k4i >= NPW).any() ): continue
                    k4  = kGRID[k4i[0],k4i[1],k4i[2],:]
                    dot = np.dot( k2 - k4, k2 - k4 )
                    if ( dot == 0.0 ): continue
                
                    FACTOR_EXCH = 4 * np.pi / dot
                    D34 = get_1RDM( C, k3i[0] * NPW**2 + k3i[1] * NPW + k3i[2], k4i[0] * NPW**2 + k4i[1] * NPW + k4i[2] )
                    V_ee_K += FACTOR_EXCH * D34
                    
    return V_ee_J - 0.5 * V_ee_K

def get_V_ee_SLOW( k1, k2, C, kGRID ):
    """
    Chemists' Notation:
    J = [ 12 | 34 ] D_{34}
    K = [ 13 | 24 ] D_{34}
    Condition: k3 - k4 = k2 - k1 --> k3 = k2 - k1 + k4
    """
    ### Coulomb Term ###
    V_ee_J = 0.0 + 0.0j
    dot = np.dot( k2 - k1, k2 - k1 )
    if dot != 0.0:
        one_over_r = 4 * np.pi / np.dot( k2 - k1, k2 - k1 )
        for k4i, k4 in enumerate(kGRID):
            for k3i, k3 in enumerate(kGRID):
                if ( k3 - k4 == k2 - k1 ).all():
                    D34 = get_1RDM( C[:,:n_elec_alpha], k3i, k4i )
                    V_ee_J += D34 * one_over_r
    
    ### Exchange Term ###
    V_ee_K = 0.0 + 0.0j
    for k4i, k4 in enumerate(kGRID):
        for k3i, k3 in enumerate(kGRID):
            if ( k3 + k4 == k2 + k1 ).all() & ( np.dot( k4 - k2, k4 - k2 ) != 0.0 ).all():
                D34 = get_1RDM( C[:,:n_elec_alpha], k3i, k4i )
                K   = 4 * np.pi / np.dot( k4 - k2, k4 - k2 )
                V_ee_K += K * D34  
    
    return V_ee_J - 0.500 * V_ee_K

def get_V_NN( kGRID ):
    #RdotK = np.einsum("Rd,Kd->RK", coords[:,:] * kGRID[:,:])
    #G     = np.einsum("RK->K", np.exp( 1j * RdotK[:,:] ) )  # Basically a structure factor
    #E_NN  = 4 * np.pi * np.sum( charges[:] * charges[:] / np.abs(kGRID[:])**2 * G[:] )

    """
    DONE IN REAL-SPACE FOR TESTING. NEED TO BE DONE IN RECIPROCAL SPACE.
    """
    E_NN  = [charges[A] * charges[B] / np.linalg.norm( coords[A,:] - coords[B,:] ) for A in range(len(charges)) for B in range(len(charges)) if A != B]
    E_NN  = 0.5 * np.sum( E_NN ) # 0.5 for double counting
    return E_NN

def get_initial_guess():
    # Assume neutral system -- Fill n_elec_alpha orbitals
    C    = np.zeros( (NPW**3, n_elec_alpha), dtype=np.complex128)
    C[:] = np.exp( 1j * np.random.uniform( size=(NPW**3, n_elec_alpha)) * 2 * np.pi )
    return C / np.linalg.norm( C )

#@njit
def __get_FOCK( C, kGRID ):
    """
    FC_ai  = \\sum_b F_ab C_bi -- LHS of eigenvalue equation F C = E C
    """
    FC = np.zeros( (NPW**3, n_elec_alpha), dtype=np.complex128 )
    for k1i, k1 in enumerate(kGRID):
        T0 = time()
        print( "k1i: ", k1i, "of", len(kGRID) )
        T_e = get_T_e( k1 )
        for k2i in range(k1i+1,NPW**3):
            k2 = kGRID[k2i]
            #print( "k2i: ", k2i, "of", len(kGRID[k1i+1:]) )
            
            
            #V_NN = get_V_NN( k1, k2 )
            V_eN      = get_V_eN( k1, k2 )
            V_ee      = get_V_ee(k1i, k2i, k1, k2, C, kGRID )
            #V_ee_SLOW = get_V_ee_SLOW(k1, k2, C, kGRID )
            #if ( abs(V_ee_SLOW) > 0.0  ):
            #print( k1i, k2i )
            #print( V_ee, np.conj(get_V_ee( k2i, k1i, k2, k1, C, kGRID )) )
            #print( V_ee_SLOW, np.conj(get_V_ee_SLOW( k2, k1, C, kGRID )) )

            #if (  np.abs(V_ee - V_ee_SLOW) > 1e-15  ):
            #    print("Not equal", k1i, k2i, V_ee, V_ee_SLOW)
            #    exit()


            tmp  = (T_e + V_eN + V_ee) * C[k2i,:]
            FC[k1i,:] += ( tmp + np.conj(tmp) ) # * 2 * 0.5 for double counting and average of .T
        print("Time: ", time() - T0)
    #exit()
    return FC

def build_solve_C_Fock_C_matrix( C, kGRID ):
    """
    F_ab   = (T_e + V_eN)_ab + (V_ee)_ab[C], [] -- Depenenence on C
    FC_ai  = \\sum_b F_ab C_bi -- LHS of eigenvalue equation F C = E C
    CFC_ji = \\sum_a C_{aj}.conj() FC_{ai} -- Fock matrix in MO basis (i.e. smaller than kGRID)
    """
    FC   = __get_FOCK( C, kGRID ) 
    #print( FC )
    CFC = np.einsum( "aj,ai->ji", C.conj(), FC )
    #print( CFC )
    E, U = np.linalg.eigh( CFC )
    print( E )
    #print( U )
    C    = np.einsum( "ai,iN->aN", C, U ) # Transform out of old "i" MO basis to new "N" MO basis and keep plane wave basis
    return C

def main():
    get_Globals()
    kGRID = get_kGRID()
    #E_NN  = get_V_NN( kGRID )
    C_MO  = get_initial_guess()
    for i in range(10):
        print( "Iteration: ", i )
        C_MO = build_solve_C_Fock_C_matrix( C_MO, kGRID )
        #print( C_MO.shape )
        #print( C_MO[:,0] )

if ( __name__ == "__main__" ):
    main()