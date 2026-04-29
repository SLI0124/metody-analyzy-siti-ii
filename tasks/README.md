# Zadání k úlohám

Tenhle přehled převádí témata z přednášek do čitelného zadání pro jednotlivé úlohy v `tasks`. Nejde o popis kódu po řádcích, ale o to, co se mělo udělat, co dané cvičení skutečně řeší a kde je řešení rozšířené oproti tomu, co bylo přímo na slajdech.

## Task 01

Tady je cílem postavit vlastní práci s velkou jednoduchou sítí nad řídkou reprezentací. Zadání z přednášky směřuje k tomu, aby se síť nebrala jako hotová struktura z knihovny, ale aby se implementovala DoK sparse matrix a nad ní se spočítaly základní vlastnosti sítě: průměrný a maximální stupeň, rozdělení stupňů, clustering effect, rozdělení clustering coefficientu podle stupně, průměrný a maximální počet společných sousedů. Součástí zadání je i paralelizace výpočtů a měření času.

Tohle cvičení zpracovává tři různé sítě, konkrétně Facebook, YouTube a proteinovou síť, a pro každou ukládá tabulky i grafy. Výstup tedy není jen jednorázový výpočet, ale srovnatelná analýza více datasetů.

**Požadavek navíc oproti přednášce:** řešení je rozšířené o více datasetů najednou a o systematické ukládání dílčích i souhrnných výsledků do CSV a obrázků.

## Task 02

Zadání z přednášky je zaměřené na časově proměnnou síť odvozenou z DBLP. Má vzniknout program, který ze zdrojových dat vytvoří časové rámce a sleduje, jak se v čase mění průměrný stupeň, průměrný vážený stupeň a průměrný clustering coefficient. Druhá část zadání chce najít simplex s nejvyšší průměrnou vahou hran.

Tohle cvičení z DBLP opravdu vytváří časové snímky podle roku, počítá požadované statistiky a hledá nejsilnější simplex. Vedle toho převádí identifikátory autorů na jména, takže výsledek není jen anonymní kombinace čísel.

**Požadavek navíc oproti přednášce:** kromě statistiky po jednotlivých rámcích se počítá i kumulativní vývoj v čase, zvlášť se vykresluje vývoj maximální průměrné váhy simplexu a průběžné výsledky se ukládají do CSV pro další použití.

## Task 03

Smyslem téhle úlohy je link prediction na jednoduchých sítích. Podle přednášky se mají implementovat všechny probírané metody založené na podobnosti a společných sousedech, aplikovat je na sítě Karate Club, Les Misérables a Dolphins, použít cross-validaci a odhad prahu a pro každou metodu spočítat výkonnostní metriky včetně precision, recall a F1.

Řešení přesně tenhle rámec drží: nad třemi malými sítěmi porovnává více link prediction metod a vyhodnocuje jejich chování v závislosti na zvoleném prahu.

**Požadavek navíc oproti přednášce:** řešení nedělá jen základní vyhodnocení, ale ještě porovnává normalizovanou a nenormalizovanou variantu skóre, vytváří srovnávací grafy pro jednotlivé metody a skládá společný souhrn přes všechny sítě.

## Task 04

Zadání z přednášky chce implementovat generátory sítí pro Barabasi-Albertův model, Link Selection Model a Copying Model, generovat sítě větší než tisíc uzlů, vizualizovat je a porovnat jejich vlastnosti, hlavně průměrný stupeň, rozdělení stupňů, clustering coefficient a clustering effect. Součástí zadání je i přidat alespoň jeden další princip a ukázat, jak změní vlastnosti výsledné sítě.

Tohle cvičení všechny tři generátory staví a pro každý model vytváří několik variant, které se potom porovnávají přes základní síťové statistiky a distribuce stupňů.

**Požadavek navíc oproti přednášce:** jako rozšíření se nepřidávají jen vnitřní hrany během růstu sítě, ale navíc se zkoumá i varianta po náhodném mazání uzlů. Nad rámec původního zadání se také ukládají exporty uzlů a hran a kreslí se další srovnávací křivky mezi variantami.

## Task 05

Tahle část přednášek je obecnější a neformuluje jedno úzké seminární zadání. Téma je vícevrstvá síť, její modely a míry, zejména actor measures, layer measures, vizualizace, community detection, edge patterns a dynamické procesy. Praktické zadání se tedy musí z těchto okruhů konkretizovat.

V tomhle cvičení je zvolena multiplexní sociální síť CS Aarhus a úloha je postavená jako základní analytický rozbor vícevrstvé sítě. Počítají se míry pro aktéry i vrstvy, například degree po vrstvách, agregovaný degree, variabilita mezi vrstvami, neighborhood, connective redundancy a exclusive neighborhood. Výsledek zároveň obsahuje přehledové vizualizace vrstev, korelace mezi vrstvami a překryv hran.

**Požadavek navíc oproti přednášce:** konkrétní výběr datasetu i sada měr jsou už rozhodnutí navíc, protože přednáška v tomhle bodě ještě nedává přesný seznam kroků. Navíc se dělají i kombinace vrstev a detailnější vizualizace top aktérů a překryvu mezi vrstvami.

## Task 06

Tady už je seminární zadání z přednášky konkrétní: vybrat jednu multilayer síť, spočítat pro každého aktéra relevance measures ze slajdů, implementovat random walk procesy a z nich odvodit occupation centrality a nakonec udělat weighted i unweighted flattening vybrané sítě.

Tohle cvičení na zvolené síti CS Aarhus přesně tyto tři části řeší. Pro jednotlivé aktéry počítá relevance a exclusive relevance, simuluje random walk napříč vrstvami a vytváří neohodnocenou i ohodnocenou zploštělou síť.

**Požadavek navíc oproti přednášce:** vedle samotných měr řešení ještě porovnává pravděpodobnosti z random walku s degree rozdělením, exportuje flattening do formátů vhodných pro další vizualizaci a ukládá tabulku se srovnáním původní multilayer sítě a obou flattening variant.

## Task 07

Zadání z přednášky chce připravit vizualizaci sítě používané v seminárních úlohách a potom provést community detection nad vícevrstvou sítí. Porovnání má proběhnout jak na jednotlivých vrstvách, tak na flatteningu všech vrstev nebo vybraných kombinací.

Tohle cvičení připravuje několik typů pohledů na CS Aarhus síť: řezy jednotlivými vrstvami, augmentovaný flattened pohled a vizualizace komunit. Community detection se pak porovnává mezi různými způsoby skládání vrstev a vyhodnocuje se modularita.

**Požadavek navíc oproti přednášce:** řešení nejde jen po jedné detekci komunit, ale systematicky sleduje, jak se modularita mění při postupném přidávání vrstev a u více kombinací flatteningu. Navíc vytváří souhrnné statistiky velikostí komunit a trendové grafy.

## Task 08

Smyslem téhle úlohy je přenést link prediction do vícevrstvého prostředí. Zadání z přednášky říká udělat experiment s link prediction v multilayer síti a vyhodnotit výsledky stejným způsobem jako u jednoduchých sítí.

Tohle cvičení bere jednotlivé vrstvy jako cílové sítě pro predikci a porovnává, jak fungují klasické similarity metody samy o sobě a jak se změní, když se do predikce přidá informace z ostatních vrstev.

**Požadavek navíc oproti přednášce:** nad základním link prediction experimentem se ještě těží association rules mezi vrstvami a ty se používají jako dodatečné cross-layer features. Výsledkem je explicitní srovnání baseline přístupu proti rozšířenému přístupu po vrstvách i po metodách.

## Task 09

Zadání ke cvičení chce na malé síti nasimulovat alespoň tři modely šíření nákazy, konkrétně SI a ještě dva další, a k tomu i šíření vlivu. Každá simulace má být vizualizovaná podobně jako v prezentaci. Druhá část chce řešit maximalizaci šíření vlivu na velkých sítích z předchozích prezentací a obě části statisticky vyhodnotit.

Tohle cvičení na malé síti Karate Club simuluje SI, SIS, SIR a Independent Cascade, ukládá průběhy stavů i snímky vývoje. Na velké síti se pak soustředí na Facebook síť, kde porovnává více seedovacích strategií, více velikostí seed setu a více pravděpodobností aktivace pro šíření vlivu.

**Požadavek navíc oproti přednášce:** rozšířením je systematický benchmark seedovacích strategií a test více hodnot aktivační pravděpodobnosti. Zároveň je fér říct, že tohle konkrétní řešení nepracuje se třemi velkými sítěmi, ale soustředí se detailně na jednu velkou síť a vyhodnocuje ji důkladněji.
