import requests
from collections import Counter
from typing import Dict, List, Tuple
import random
import math
from copy import deepcopy

def create_chunks(ciphertext: str, chunk_size: int = 8) -> List[str]:
    """
    Découpe le texte chiffré en morceaux de 8 bits.
    """
    return [ciphertext[i:i+chunk_size] for i in range(0, len(ciphertext), chunk_size)]

def analyze_ciphertext(ciphertext: str) -> Tuple[Counter, List[str], List[str]]:
    """
    Analyse le texte chiffré pour extraire les patterns et les doubles consécutifs.
    """
    chunks = create_chunks(ciphertext)
    pattern_freq = Counter(chunks)
    
    # Identifier les patterns qui apparaissent en double consécutif
    consecutive_doubles = []
    for i in range(len(chunks)-1):
        if chunks[i] == chunks[i+1] and chunks[i] not in consecutive_doubles:
            consecutive_doubles.append(chunks[i])
    
    return pattern_freq, chunks, consecutive_doubles

class FrequencyAnalyzer:
    def __init__(self):
        self.corpus = self.load_corpus()
        self.char_frequencies, self.bigram_frequencies = self.analyze_corpus(self.corpus)
        
        # Définition des règles pour les lettres doublées
        self.vowels = set('aeiouàâäéèêëîïôöùûü')
        self.consonants = set('bcdfghjklmnpqrstvwxz')
        self.double_letter_rules = {
            'tt': self.vowels | {'r'},
            'ss': self.vowels | {'p'},
            'ff': self.vowels,
            'll': self.vowels,
            'rr': self.vowels | {'t'}
        }
        
        # Paires de lettres interdites
        self.forbidden_pairs = {
            'qx', 'wq', 'kq', 'qz', 'qk', 'qw', 'qy', 'jx', 'jz', 'jq', 
            'wx', 'wz', 'wk', 'zx', 'ct', 'gm', 'lg', 'ks', 'rs', 'rt', 
            'xt', 'mc', 'mj', 'mv', 'çb', 'çl', 'çs', 'çq', 'çh', 'ïa', 
            'ïo', 'jj', 'hhj', 'zq'
        }
        # Suite de la classe FrequencyAnalyzer
    def get_most_common(self) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        sorted_chars = sorted(
            self.char_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        sorted_bigrams = sorted(
            self.bigram_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_chars, sorted_bigrams

    def is_valid_sequence(self, sequence: str) -> bool:
      #Verifie que ce nest pas une paire interdite
        for i in range(len(sequence)-1):
            if sequence[i:i+2] in self.forbidden_pairs:
                return False
      #Verifie que 3 consonnes ou 3 voyelles ne se repetent pas
        for i in range(len(sequence)-2):
            if all(c in self.consonants for c in sequence[i:i+3]):
                return False
            if all(c in self.vowels for c in sequence[i:i+3]):
                return False

        for i in range(len(sequence)-2):
            if sequence[i] == sequence[i+1]:
                double = sequence[i:i+2]
                next_char = sequence[i+2] if i+2 < len(sequence) else None
                if double in self.double_letter_rules and next_char:
                    if next_char not in self.double_letter_rules[double]:
                        return False
        return True

    def is_valid_double_letter_sequence(self, double: str, next_char: str) -> bool:
        #Verifie que les chunks qui se repetent sont des lettres qui se doublent en francais
        if double in self.double_letter_rules:
            return next_char in self.double_letter_rules[double]
        return True

    def load_corpus(self) -> str:
        #Importe un ensemble de livre pour former un corpus
        urls = [
            "https://www.gutenberg.org/files/13846/13846-0.txt",
            "https://www.gutenberg.org/files/4650/4650-0.txt"
        ]
        corpus = ""
        for url in urls:
            try:
                response = requests.get(url)
                response.raise_for_status()
                corpus += response.text.upper()
            except Exception as e:
                print(f"Erreur lors du chargement de {url}: {e}")
        return corpus

    def analyze_corpus(self, corpus: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        clean_corpus = ''.join(c for c in corpus.upper() if c.isalpha() or c.isspace())
        
        char_freq = Counter(clean_corpus)
        total_chars = len(clean_corpus)
        #On compte la frequence des caracteres et des bigrammes dans le corpus pour former un dictionnaire de frequence
        char_frequencies = {k: v/total_chars for k, v in char_freq.items()}
        
        bigrams = [''.join(pair) for pair in zip(clean_corpus[:-1], clean_corpus[1:])]
        bigram_freq = Counter(bigrams)
        total_bigrams = len(bigrams)
        bigram_frequencies = {k: v/total_bigrams for k, v in bigram_freq.items()}
        
        return char_frequencies, bigram_frequencies

def create_initial_mapping(cipher_patterns: Counter, analyzer: FrequencyAnalyzer, consecutive_doubles: List[str], chunks: List[str]) -> Dict[str, str]:
    """
    Crée un mapping initial en deux phases:

    1. Mapper les patterns les plus fréquents aux lettres simples
    2. Mapper les patterns restants aux bigrammes
    """
    sorted_chars, sorted_bigrams = analyzer.get_most_common()
    sorted_cipher = sorted(cipher_patterns.items(), key=lambda x: x[1], reverse=True)
    mapping = {}
    used_chars = set()

    # Phase 1: Mapping des lettres simples
    print("\nPhase 1: Mapping des lettres simples")
    available_chars = [char for char, _ in sorted_chars if char.isalpha()] 

    # D'abord traiter les doubles lettres
    for pattern in consecutive_doubles:
        for double_letter in 'tsflr':
            if double_letter not in used_chars and double_letter in available_chars:  # Vérifier si la lettre est disponible
                mapping[pattern] = double_letter
                used_chars.add(double_letter)
                available_chars.remove(double_letter)
                print(f"Double lettre mappée: {pattern} -> {double_letter}")
                break

    # Ensuite mapper les patterns les plus fréquents aux lettres simples restantes
    for pattern, freq in sorted_cipher:
        if pattern not in mapping and available_chars:
            char = available_chars[0]  # Prendre le premier caractère disponible
            mapping[pattern] = char
            used_chars.add(char)
            available_chars.remove(char)
            print(f"Lettre simple mappée: {pattern} -> {char}")

    # Phase 2: Mapping des bigrammes
    unmapped_patterns = [p for p, _ in sorted_cipher if p not in mapping]
    available_bigrams = [bigram for bigram, _ in sorted_bigrams]

    for pattern in unmapped_patterns:
        # Chercher un bigramme valide
        for bigram in available_bigrams:
            # Vérifier que le bigramme n'utilise pas de lettres déjà utilisées
            if not any(c in used_chars for c in bigram):
                is_valid = True
                # Vérifier que le bigramme respecte les règles du français
                for i, chunk in enumerate(chunks[:-2]):
                    if chunk == pattern:
                        prev_char = mapping.get(chunks[i-1], '') if i > 0 else ''
                        next_char = mapping.get(chunks[i+1], '')
                        sequence = prev_char + bigram + next_char
                        if sequence and not analyzer.is_valid_sequence(sequence):
                            is_valid = False
                            break

                if is_valid:
                    mapping[pattern] = bigram
                    used_chars.update(bigram)
                    available_bigrams.remove(bigram)
                    print(f"Bigramme mappé: {pattern} -> {bigram}")
                    break

        # Si aucun bigramme valide n'est trouvé, attribuer un bigramme aléatoire
        if pattern not in mapping and available_bigrams:
            bigram = random.choice(available_bigrams)
            mapping[pattern] = bigram
            used_chars.update(bigram)
            available_bigrams.remove(bigram)
            print(f"Bigramme aléatoire mappé: {pattern} -> {bigram}")
    # Afficher les patterns non mappés
    unmapped = [p for p, _ in sorted_cipher if p not in mapping]
    if unmapped:
        print("\nPatterns non mappés:")
        for p in unmapped[:10]:
            print(f"  {p}: {cipher_patterns[p]} occurrences")

    return mapping

def modify_mapping(mapping: Dict[str, str], analyzer: FrequencyAnalyzer, consecutive_doubles: List[str], chunks: List[str]) -> Dict[str, str]:
    # Créer une copie profonde du mapping pour éviter de modifier l'original
    new_mapping = deepcopy(mapping)
    
    def is_valid_swap(key1: str, key2: str, val1: str, val2: str) -> bool:
        # Créer un mapping temporaire avec l'échange de valeurs
        temp_mapping = deepcopy(new_mapping)
        temp_mapping[key1] = val2
        temp_mapping[key2] = val1
        
        for key in [key1, key2]:
            for i, chunk in enumerate(chunks[:-2]):
                if chunk == key:
                    # Récupérer les chunks précédent et suivant
                    prev_chunk = chunks[i-1] if i > 0 else None
                    next_chunk = chunks[i+1] if i+1 < len(chunks) else None
                    
                    # Construire la séquence à vérifier
                    sequence = []
                    if prev_chunk and prev_chunk in temp_mapping:
                        sequence.append(temp_mapping[prev_chunk])
                    sequence.append(temp_mapping[key])
                    if next_chunk and next_chunk in temp_mapping:
                        sequence.append(temp_mapping[next_chunk])
                    
                    # Vérifier si la séquence respecte les règles linguistiques
                    if sequence and not analyzer.is_valid_sequence(''.join(sequence)):
                        return False
                    
                    # Vérifier les règles pour les lettres doublées
                    if key in consecutive_doubles:
                        if next_chunk and next_chunk in temp_mapping:
                            if not analyzer.is_valid_double_letter_sequence(
                                temp_mapping[key]*2, 
                                temp_mapping[next_chunk]
                            ):
                                return False
        return True

    # Effectuer jusqu'à 20 tentatives de modification du mapping
    for _ in range(20):
        # Choisir une stratégie aléatoire de modification
        strategy = random.random()
        
        # Stratégie 1 : Échanger des patterns de lettres doublées
        if strategy < 0.3 and consecutive_doubles:
            available_doubles = [k for k in consecutive_doubles if k in new_mapping]
            if len(available_doubles) >= 2:
                key1, key2 = random.sample(available_doubles, 2)
                val1, val2 = new_mapping[key1], new_mapping[key2]
                if is_valid_swap(key1, key2, val1, val2):
                    new_mapping[key1], new_mapping[key2] = val2, val1
                    break
        
        # Stratégie 2 : Échanger des patterns de lettres simples
        elif strategy < 0.7:
            single_chars = [k for k, v in new_mapping.items() if len(v) == 1 and k not in consecutive_doubles]
            if len(single_chars) >= 2:
                key1, key2 = random.sample(single_chars, 2)
                val1, val2 = new_mapping[key1], new_mapping[key2]
                if is_valid_swap(key1, key2, val1, val2):
                    new_mapping[key1], new_mapping[key2] = val2, val1
                    break
        
        # Stratégie 3 : Échanger des patterns de bigrammes
        else:
            bigrams = [k for k, v in new_mapping.items() if len(v) == 2]
            if len(bigrams) >= 2:
                key1, key2 = random.sample(bigrams, 2)
                val1, val2 = new_mapping[key1], new_mapping[key2]
                if is_valid_swap(key1, key2, val1, val2):
                    new_mapping[key1], new_mapping[key2] = val2, val1
                    break
    
    # Retourner le nouveau mapping potentiellement modifié
    return new_mapping
def decode_text(chunks: List[str], mapping: Dict[str, str]) -> str:
    """
    Décode le texte en utilisant le mapping fourni.
    
    Args:
        chunks: Liste des morceaux de texte chiffré
        mapping: Dictionnaire de correspondance entre patterns et lettres/bigrammes
    
    Returns:
        str: Texte décodé
    """
    decoded = []
    i = 0
    while i < len(chunks):
        current_chunk = chunks[i]
        
        # Vérifier si c'est un pattern doublé
        if (i < len(chunks) - 1 and 
            chunks[i] == chunks[i+1] and 
            mapping.get(current_chunk, '') in 'tsflr'):
            # C'est une lettre doublée
            letter = mapping[current_chunk]
            decoded.append(letter * 2)
            i += 2
        else:
            # Pattern normal (caractère unique ou bigramme)
            value = mapping.get(current_chunk, '□')
            decoded.append(value)
            i += 1
    
    return ''.join(decoded)

def score_text(decoded_text: str, analyzer: FrequencyAnalyzer) -> float:
    """
    Calcule un score pour le texte décodé basé sur plusieurs critères.
    
    Args:
        decoded_text: Texte décodé à évaluer
        analyzer: Instance de FrequencyAnalyzer contenant les statistiques de référence
    
    Returns:
        float: Score calculé
    """
    score = 0.0
    text_length = len(decoded_text)
    
    # Score basé sur les fréquences des caractères
    char_freq = Counter(decoded_text)
    for char, count in char_freq.items():
        if char in analyzer.char_frequencies:
            expected_freq = analyzer.char_frequencies[char]
            actual_freq = count / text_length
            score += (1 - abs(expected_freq - actual_freq)) * 100

    # Score basé sur les fréquences des bigrammes
    bigrams = [''.join(pair) for pair in zip(decoded_text[:-1], decoded_text[1:])]
    bigram_freq = Counter(bigrams)
    for bigram, count in bigram_freq.items():
        if bigram in analyzer.bigram_frequencies:
            expected_freq = analyzer.bigram_frequencies[bigram]
            actual_freq = count / (text_length - 1)
            score += (1 - abs(expected_freq - actual_freq)) * 200

    # Bonus pour les lettres doublées correctement placées
    for i in range(len(decoded_text)-2):
        if decoded_text[i] == decoded_text[i+1]:
            double = decoded_text[i:i+2]
            next_char = decoded_text[i+2] if i+2 < len(decoded_text) else ''
            if (double[0]*2 in analyzer.double_letter_rules and 
                next_char in analyzer.double_letter_rules[double[0]*2]):
                score += 50

    # Pénalités
    # Pour les caractères non décodés
    score -= decoded_text.count('□') * 50
    
    # Pour les séquences interdites
    for i in range(len(decoded_text)-1):
        if decoded_text[i:i+2] in analyzer.forbidden_pairs:
            score -= 100

    # Pour les triples consonnes/voyelles
    for i in range(len(decoded_text)-2):
        sequence = decoded_text[i:i+3]
        if (all(c in analyzer.consonants for c in sequence) or 
            all(c in analyzer.vowels for c in sequence)):
            score -= 100

    return score

def print_text_statistics(text: str, analyzer: FrequencyAnalyzer) -> None:
    """
    Affiche des statistiques sur le texte décodé.
    
    Args:
        text: Texte décodé à analyser
        analyzer: Instance de FrequencyAnalyzer
    """
    print(f"Longueur totale: {len(text)}")
    print(f"Caractères non décodés: {text.count('□')}")
    
    # Analyse des lettres doublées
    doubles = {}
    for i in range(len(text)-1):
        if text[i] == text[i+1]:
            double = text[i:i+2]
            if double not in doubles:
                doubles[double] = 0
            doubles[double] += 1
    
    print("\nLettres doublées trouvées:")
    for double, count in doubles.items():
        print(f"{double}: {count} occurrences")
    
    # Vérification des règles
    violations = []
    for i in range(len(text)-2):
        sequence = text[i:i+3]
        if all(c in analyzer.consonants for c in sequence):
            violations.append(f"Triple consonne: {sequence}")
        elif all(c in analyzer.vowels for c in sequence):
            violations.append(f"Triple voyelle: {sequence}")
    
    if violations:
        print("\nViolations trouvées:")
        for v in violations[:10]:  # Afficher les 10 premières violations
            print(v)

def decrypt(ciphertext: str, max_iterations: int = 10000) -> Tuple[str, Dict[str, str]]:
    analyzer = FrequencyAnalyzer()
    pattern_freq, chunks, consecutive_doubles = analyze_ciphertext(ciphertext)
    
    print("Analyse initiale:")
    print(f"Nombre de patterns uniques: {len(pattern_freq)}")
    print(f"Nombre de patterns doublés: {len(consecutive_doubles)}")
    
    best_mapping = create_initial_mapping(
        cipher_patterns=pattern_freq,
        analyzer=ana  lyzer,
        consecutive_doubles=consecutive_doubles,
        chunks=chunks 
    )
    
    best_decoded = decode_text(chunks, best_mapping)
    best_score = score_text(best_decoded, analyzer)
    
    print("\nDébut de l'optimisation:")
    print(f"Score initial: {best_score}")
    print(f"Exemple de texte initial: {best_decoded[:100]}")
    
    no_improvement = 0
    temperature = 1.0
    
    for iteration in range(max_iterations):
        temperature *= 0.999
        
        new_mapping = modify_mapping(best_mapping, analyzer, consecutive_doubles, chunks)
        new_decoded = decode_text(chunks, new_mapping)
        new_score = score_text(new_decoded, analyzer)
        
        score_diff = new_score - best_score
        if score_diff > 0 or random.random() < math.exp(score_diff / temperature):
            best_score = new_score
            best_mapping = new_mapping
            best_decoded = new_decoded
            no_improvement = 0
            
            if score_diff > 0:
                print(f"\nItération {iteration + 1}:")
                print(f"Nouveau score: {best_score}")
                print(f"Exemple: {best_decoded[:100]}")
        else:
            no_improvement += 1
        
        if no_improvement >= 1000 and iteration > 5000:
            print("\nArrêt anticipé: pas d'amélioration significative")
            break
    
    return best_decoded
