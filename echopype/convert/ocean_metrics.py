# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:27:36 2019

@author: Sven Gastauer
@licence: GPLv3 or higher
@credits: Yoann Ladroit, Dezhang Chu, Martin J. Cox
@maintainer: Sven Gastauer
@status: development

Currently contains computations of :
    - Abosrption
    - Water Density
    - Sound speed in water

"""
import numpy as np
class Ocean_Metrics():
    
    ''''''''''''''''''''''
    Compute absorption
    
    Returns the Absorption coefficient [dB/km] for
    the given acoustic frequency (f [kHz]), salinity
    (S, [ppt]), temperature (T, [degC]), and depth
    (D, [m]).
    
    Using formula of Doonan or Francois and Garrison
    
    '''''''''''''''''''''
    
    def seawater_absorption(self,f,S,T,D,pH=None, method='Doonan'):
        '''
         Returns the Absorption coefficient [dB/km] for
         the given acoustic frequency (f [kHz]), salinity
         (S, [ppt]), temperature (T, [degC]), and depth
         (D, [m]).
        
         Note that the salinity units are ppt, not the more
         modern psu. See the comment on page 2 of Doonan et al.
         for a discussion of this. For most purposes, using psu
         values in place of ppt will make no difference to the
         resulting alpha value.
        
         
         By default function implements the formula given in
         Doonan, I.J.; Coombs, R.F.; McClatchie, S. (2003).
         The Absorption of sound in seawater in relation to
         estimation of deep-water fish biomass. ICES
         Journal of Marine Science 60: 1-9.
        
         Note that the paper has two errors in the formulae given
         in the conclusions.
        
         Optional argument 'method', allows the user to specify the
         Absorption formula to use. Default is Doonan etal (2003)
         other possibilities are:
         'doonan'  - Doonan et al (2003)
         'fandg'   - Francois & Garrison (1982)
         
         Based on and modified from Matlab code written  by 
         Gavin Macaulay, August 2003 and Yoann Ladroit 2015
         
        '''
        def get_method():
            if method == 'fandg':
                m = 'Francois and Garrison (1982)'
            elif method == 'Doonan':
                m = 'Doonan et al (2003)'
            return m
        
        print("Computing absorption, using %s" % get_method())
        if method == 'Doonan':
           #Check if temperature is higher then 20 or the frequency is above 120 kHz,
           #outside of Doonan's validity...IF so change method to Francois and Garrison
            if T > 20 or f < 10 or f>120:
                method = 'fandg'
                print("Switching method to %s" % get_method())
        
            c = 1412 + 3.21 * T + 1.19 * S + 0.0167 * D
            A2 = 22.19 * S * (1.0 + 0.017 * T)
            f2 = 1.8 * 10**(7.0 - 1518/(T+273))
            #f2 = 1.8 * 10**7.0 * np.exp( -1818/(T+273.1))
            P2 = np.exp(-1.76e-4*D)
            A3 = 4.937e-4 - 2.59e-5*T + 9.11e-7*T*T - 1.5e-8*T*T*T
            P3 = 1.0 - 3.83e-5 * D + 4.9e-10 * D * D
            alpha = A2 * P2 * f2 * f * f / ( f2 * f2 + f * f ) / c + A3 * P3 * f * f
        
        elif method == 'fandg':
            if pH is None:
                if S > 10: #assume sea water
                    pH = 8
                else: #assume fresh water
                    pH = 7
                
            c = 1412 + 3.21 * T + 1.19 * S + 0.0167 * D
            A1 = 8.86 * (10** ( 0.78 * pH - 5)) / c
            A2 = 21.44 * S * (1 + 0.025 * T) / c
            A3 = 4.937e-4 - 2.59e-5*T + 9.11e-7 * T * T - 1.5e-8 * T * T * T
            if T > 20:
                A3 = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T * T - 6.5e-10 * T * T * T
            else:
                A3 = 3.964e-4;
            f1 = 2.8 * np.sqrt( S / 35) * (10**( 4 - 1245 / ( T + 273 )))
            f2 = (8.17 * 10 ** (8 - 1990 / (T + 273))) / (1 + 0.0018 * ( S - 35 ))
            P2 = 1.0 - 1.37e-4 * D + 6.2e-9 * D * D
            P3 = 1.0 - 3.83e-5 * D + 4.9e-10 * D * D
            alpha = f **2 * (A1 * f1 / ( f1 **2 + f **2 ) + A2 * P2 * f2 / ( f2 **2 + f **2 ) + A3 * P3)
        print("Absorption is %.2f dB/km for %i kHz" %(alpha,f))
        return(alpha)
        
        
        
    ''''''''''''''''''''''
    Compute water density
    
    Seawater Density according to UNESCO formula
    
    @description UNESCO (1981) Tenth report of the joint panel on
                 oceanographic tables and standards. UNESCO Technical
                 Papers in Marine Science, Paris, 25p
                 
    '''''''''''''''''''''

    def _rho_smow(self, T):
        
        '''
        Standard  Mean Ocean Water (SMOW)
        @param T Temperature in degrees Celsius
        '''
        a0 = 999.842594
        a1 = 6.793953 * 10**-2
        a2 = -9.095290 * 10**-3
        a3 = 1.001685 * 10**-4
        a4 = -1.120083 * 10**-6
        a5 = 6.536332 * 10**-9

        return(a0 + a1*T + a2*T**2 + a3*T**3 + a4*T**4 + a5*T**5)

        
    def _rho_p0(self, S, T):
        '''
        Density (rho) at normal atmospheric pressure (p = 0)
        @param S Salinity in psu (~ppm)
        @param T Temperature in degrees Celsius
        '''
        b0 = 8.2449 * 10**-1
        b1 = -4.0899 * 10**-3
        b2 = 7.6438 * 10**-5
        b3 = -8.2467 * 10**-7
        b4 = 5.3875 * 10**-9
        
        c0 = -5.7246 * 10**-3
        c1 = 1.0227 * 10**-4
        c2 = -1.6546 * 10**-6
        d0 = 4.8314 * 10**-4
    
        B1 = b0 + b1 * T + b2 * T**2 + b3 * T**3 + b4 * T**4
        C1 = c0 + c1 * T + c2 * T**2
        
        return(self._rho_smow(T) + B1*S + C1*S**1.5 + d0*S**2)
               
    def _K_p0(self, S, T):
        
        '''
        compressibility at p = 0
        @param S Salinity in psu (~ppm)
        @param T Temperature in degrees Celsius
        
        '''
        
        e0 = 19652.210
        e1 = 148.4206
        e2 = -2.327105
        e3 = 1.360477 * 10**-2
        e4 = -5.155288 * 10**-5
        
        K_w = e0 + e1 * T + e2 * T**2 + e3 * T**3 + e4 * T**4
        
        f0 = 54.6746
        f1 = -0.603459
        f2 = 1.099870 * 10**-2
        f3 = -6.167 * 10**-5
        
        F1 = f0 + f1 * T + f2 * T**2 + f3 * T**3
        
        g0 = 7.944 * 10**-2
        g1 = 1.6483 * 10**-2
        g2 = -5.3009 * 10**-4
        
        G1 = g0 + g1*T + g2*T**2
        
        return(K_w + F1 * S + G1 * S**1.5)
    
    def _K(self, S,T,p):
        
        '''
        compressibility K
        @param S Salinity in psu (~ppm)
        @param T Temperature in degrees Celsius
        @param p pressure in bar
        
        '''
        
        h0 = 3.2399
        h1 = 1.43713 * 10**-3
        h2 = 1.16092 * 10**-4
        h3 = -5.77905 * 10**-7
        
        i0 = 2.838 * 10**-3
        i1 = -1.0981 * 10**-5
        i2 = -1.6078 * 10**-6
        j0 = 1.91075 * 10**-4
        
        k0 = 8.50935 * 10**-5
        k1 = -6.12293 * 10**-6
        k2 = 5.2787 * 10**-8
        
        m0 = -9.9348 * 10**-7
        m1 = 2.0816 * 10**-8
        m2 = 9.1697 * 10**-10
        
        Bw = k0 + k1*T + k2*T**2
        B2 = Bw + (m0 + m1*T + m2*T**2)*S
        
        Aw = h0 + h1*T + h2*T**2 + h3*T**3
        A1 = Aw + (i0 + i1*T + i2*T**2) * S + j0*S**1.5
        
        return(self._K_p0(S,T) + A1*p + B2*p**2)

    
    def rho_w(self, S, T, p=0):
        '''
        Seawater Density according to UNESCO formula
        @description  UNESCO (1981) Tenth report of the joint panel on
        oceanographic tables and standards. UNESCO Technical
        Papers in Marine Science, Paris, 25p
        
        @params S: Salinity in psu (which is +/- equal to ppm)
        @units S: psu/ppm
        @params T: Temperature in degrees
        @units T: degree Celsius
        @params p: pressure in Bar
        @units p: Bar
        @examples
        rho(S=35,T=0.5,p=10)
        rho(8,10) #Should be 1005.94659
        '''         
        if p == 0: 
            rho = self._rho_p0(S,T)
        else: 
            rho = self._rho_p0(S,T) / (1 - p / self._K(S,T,p) ) 
        print("Water density rho = %.2f kg/m3" %rho)
        return(rho)
        
    ''''''''''''''''''''''
    Compute sound velocity in water
    
    
                 
    '''''''''''''''''''''
    

    def _c_Mackenzie1981(self, D,S,T):
        '''
        Sound speed according to Mackenzie et al. (1981)
        @description Calculate speed of sound in seawater based on MacKenzie (1981)
        The empirical equation generally holds validity for a temperature range between 2 and 30 degrees Celsius, Salinities between 25 and 40 parts per thousand and a depth range between 0 and 8000 m
        @source Mackenzie, K. V. (1981).
        Nine‐term equation for sound speed in the oceans. The Journal of the Acoustical Society of America, 70(3), 807-812.
        http://asa.scitation.org/doi/abs/10.1121/1.386920
        @references Mackenzie, K. V. (1981).
        Nine‐term equation for sound speed in the oceans. The Journal of the Acoustical Society of America, 70(3), 807-812.
        @param D Depth in meters
        @param S Salinity in parts per thousands
        @param T Temperature in degrees Celsius
        '''
        
        c = 1448.96 + 4.591*T - 5.304 * 10**-2*(T**2) + 2.374 * (10**-4)*(T**3) + \
        1.340 * (S-35) + 1.630 * (10**-2)*D + 1.675 * (10**-7)*(D**2) - \
        1.025 * (10**-2)*T*(S - 35) - 7.139 * (10**-13)*T*(D**3)
        
        return(c)



    def _c_Coppens1981(self, D, S, T):
        '''
        @title Sound speed according to Coppens et al. (1981)
        @description Calculates speed of sound in seawater based on Coppens (1981)
        The empirical equation generally holds validity for a temperature range between 0 and 35 degrees Celsius, Salinities between 0 and 45 parts per thousand and a depth range between 0 and 4000 m
        @source Coppens, A. B. (1981).
        Simple equations for the speed of sound in Neptunian waters. The Journal of the Acoustical Society of America, 69(3), 862-863.
        http://asa.scitation.org/doi/abs/10.1121/1.385486
        @references  Coppens, A. B. (1981).
        Simple equations for the speed of sound in Neptunian waters. The Journal of the Acoustical Society of America, 69(3), 862-863.
        @param D Depth in meters
        @param S Salinity in parts per thousands
        @param T Temperature in degrees Celsius
        @examples
        c_Coppens1981(D=100, S=35, T=10)
        '''
        
        t = T/10
        D = D/1000
        c0 = 1449.05 + 45.7*t - 5.21*(t**2)  + 0.23*(t**3)  + (1.333 - 0.126*t + 0.009*(t**2)) * (S - 35)
        c = c0 + (16.23 + 0.253*t)*D + (0.213-0.1*t)*(D**2)  + (0.016 + 0.0002*(S-35))*(S- 35)*t*D
        return(c)



    def _c_Leroy08(self, D, S, T, lat=None):
        
        ''' 
        @title Compute speed of sound according to Leroy et al. (2008)
        @description Returns the sound speed according to Leroy et al (2008). This "newer" equation should solve the sound speed within 0.2 m/s for all seas, including the Baltic and Black sea, based on Temperature, Salinity and Latitude. Exceptions are some seas with anomalities close to the bottom. The equation was specifically designed to be used in marine acoustics.
        @param D Depth in m
        @param S Salinity in parts per thousand
        @param T Temperature in degrees Celsius
        @param lat Latitude in degrees
        @source Leroy, C. C., Robinson, S. P., & Goldsmith, M. J. (2008).
        A new equation for the accurate calculation of sound speed in all oceans. The Journal of the Acoustical Society of America, 124(5), 2774-2782.
        http://asa.scitation.org/doi/abs/10.1121/1.2988296
        @references Leroy, C. C., Robinson, S. P., & Goldsmith, M. J. (2008).
        A new equation for the accurate calculation of sound speed in all oceans. The Journal of the Acoustical Society of America, 124(5), 2774-2782.
        @examples
        # TABLE III in Leroy et al. (2008)
        # Common oceans, lat = 30°, P= 80 MPa Z= 7808.13 m, S= 34.7%
        lat=30; Z=7808.13; S=34.7; T=c(1,1.5,2,2.5,3)
        c_Leroy08(Z,T,S,lat)
        # Common oceans, lat = 30°, P= 80 MPa Z= 7808.13 m, T=2 °C
        c_Leroy08(Z,T=2,S=seq(33.5,35.5,.5),lat)
        # Common oceans, = 30°, P= 5 MPa Z= 497.12 m, S= 35%
        c_Leroy08(Z=497.12,T=seq(-2,20,2),S=35,lat)
        # Common oceans, = 30°, P= 5 MPa Z= 497.12 m, T=8 °C
        c_Leroy08(Z=497.12,T=8,S=seq(33,37,1),lat)
        
        '''
               
        c = 1402.5 + 5*T - 5.44 * 10**-2*T**2 + 2.1 * 10**-4*T**3 + \
        1.33*S - 1.23 * (10**-2)*S*T+8.7*(10**-5)*S*T**2 + \
        1.56*(10**-2)*D+2.55*(10**-7)*D**2-7.3*(10**-12)*D**3+ \
        1.2*(10**-6)*D*(lat-45)-9.5*(10**-13)*T*D**3+ \
        3*(10**-7)*T**2*D+1.43*(10**-5)*S*D
        
        return(c)
            
        
    def sound_velocity(self,D,T,S,lat=None, method = "Coppens"):
        
        '''
        Compute sound velocity for given temperature, salinity and depth, with optional item latitude
        
        uses either the method according to Mackenzie, Coppens or Leroy
        
        For a description of the methods see above...
        
        '''
        if method == 'Coppens':
            c = (self._c_Coppens1981(D,S,T))
        elif method == 'Mackenzie':
            c = (self._c_Mackenzie1981(D,S,T))
        elif method == 'Leroy':
            try:
                c = self._c_Leroy08(D,S,T,lat)
            except TypeError as e:
                print("For Leroy 2008 a Latitude input is needed --- ",e)
                return
        print("According to %s -  c = %.2f m/s" %(method,c))
        return(c)
            