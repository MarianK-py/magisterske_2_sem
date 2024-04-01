module Magic11 where

-- Marian Kravec

-- Matematicky dokaz:
-- cislo je delitelne 11 ak je rozdiel suctu cifier na parnych poziciach a sucet cifier na neparnych poziciach
-- delitelny 11, kedze rozdiel medzi obdlznikovym cislom ktore ziskame v smere hodinovych ruciciek a proti
-- smeru hodinovych ruciek je len postupnost cifier na parnych poziciach (cize sumy parnych a neparnych pozicii sa nezmenia)
-- moze vo vseobecnosti uvazovat iba o cislach v smere hodinovych ruciciek, zaroven ak vytvorime obdlznik mame 4 mozne
-- obdlznikove cisla v smere hodidovych ruciciek (v zavislosti ktorou cifrou zacneme), avsak tieto cisla su cyklycke
-- cize dalsie cislo vieme vytvorit tym, ze poslednu cifru presunieme na zaciatok, kedze mame parny pocet cifier, takyto
-- presun zmeni paritu pozicie kazdej pozicie, z toho vyplyva, ze sa vymenia sucty parnych a neparnych pozicii, co znamene
-- ze ich novy rozdiel bude -1 nasobok predchadzajuceho, avsak vynásobenie cisla cislom -1 nezmeni delidelnost cislom 11
-- takze nemusime uvazovat vsetky 4 obdlznikove cisla ale staci jedno a ostatne dajú rovnaky vysledok na delitelnost 11
-- preto budeme uvazovat nad takymto obdlznikovym cislom:
-- 1 -> 2
-- A    |
-- |    v
-- 4 <- 3
-- uvazujme, te prve cislo je x, potom druhe bude x+1*i kde i je 1 alebo 2 (ak to klavesnica dovoluje)
-- tretie cislo bude x+1*i-3*j kde j je 1 alebo 2 (ak to klavesnica dovoluje) (3 preto lebo riadky maju rozdiel 3)
-- a nakoniec stvrte cislo musi byt x-3*j
-- ak teraz scitame neparne cifry dostaneme: x+x+1*i-3*j a parne cifry x+1*i+x-3*j
-- ak teraz tieto sucty od seba odpocitame dostaneme: x+x+1*i-3*j-x-1*i-x+3*j = 0
-- 0 povazujeme za delitelne 11 takze taketo obdlznikove cislo je delitelne 11,
-- kedze kazde ine obdlznikove cislo vieme z tohto cisla dostat upravami ktore nezmenia jeho
-- delitelnost 11 tak z toho vyplyva, ze kazde obdlznikove cislo je delitelne 11


-- Dokaz kodom - vyskusanim vsetkych moznosti:
kontrapriklad :: Int
kontrapriklad = najdiPriklad 1
                where
                  najdiPriklad :: Int -> Int
                  najdiPriklad cifra | cifra < 10 = if (prikladZacCif cifra 0 1) > 0 then prikladZacCif cifra 0 1 else najdiPriklad (cifra+1)
                                     | otherwise = 0

                  prikladZacCif :: Int -> Int -> Int -> Int
                  prikladZacCif cifra cislo poz | (poz == 1) && (cifra > 0) && (cifra < 10) = max (prikladZacCif (cifra+1) (((cislo)*10)+cifra) 2) (prikladZacCif (cifra+2) (((cislo)*10)+cifra) 2)
                                                | (poz == 2) && (cifra > 0) && (cifra < 10) = max (prikladZacCif (cifra-3) (((cislo)*10)+cifra) 3) (prikladZacCif (cifra-6) (((cislo)*10)+cifra) 3)
                                                | (poz == 3) && (cifra > 0) && (cifra < 10) = max (prikladZacCif (cifra-1) (((cislo)*10)+cifra) 4) (prikladZacCif (cifra-2) (((cislo)*10)+cifra) 4)
                                                | (poz == 1) && (cifra > 0) && (cifra < 10) = max (testVsetkyMoznosti (((cislo)*10)+cifra)) (testVsetkyMoznosti (((reverseNum cislo)*10)+cifra))
                                                | otherwise = 0

                  testVsetkyMoznosti :: Int -> Int
                  testVsetkyMoznosti n | not (delit11 n) = n
                                       | not (delit11 ((1000*(n `mod` 10))+(n `div` 10))) = ((1000*(n `mod` 10))+(n `div` 10))
                                       | not (delit11 ((100*(n `mod` 100))+(n `div` 100))) = ((100*(n `mod` 100))+(n `div` 100))
                                       | not (delit11 ((10*(n `mod` 1000))+(n `div` 1000))) = ((10*(n `mod` 1000))+(n `div` 1000))
                                       | otherwise = 0

                  reverseNum :: Int -> Int
                  reverseNum n = accum n 0
                                 where
                                   accum m acc | m == 0 = acc
                                               | otherwise = accum (m `div` 10) ((10*acc)+(m `mod` 10))

                  delit11 :: Int -> Bool
                  delit11 n = (n `mod` 11) == 0

