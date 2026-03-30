import array
import random

class StandardPostings:
    """ 
    Kelas dengan metode statis untuk mengubah representasi postings list
    dari list integer menjadi urutan byte (bytes) menggunakan library array.
    Metode ini tidak melakukan kompresi tambahan (hanya raw bytes).

    Asumsi: postings_list untuk sebuah term muat di memori.
    """

    @staticmethod
    def encode(postings_list):
        """
        Mengonversi postings_list menjadi aliran byte (stream of bytes).

        Args:
            postings_list (list): List dari docID (postings).

        Returns:
            bytes: Bytearray yang merepresentasikan urutan integer pada postings_list.
        """
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Mendekode postings_list dari aliran byte.

        Args:
            encoded_postings_list (bytes): Bytearray hasil dari metode encode.

        Returns:
            list: List of docID hasil dekoding.
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Mengonversi list term frequencies menjadi aliran byte.

        Args:
            tf_list (list): List of term frequencies.

        Returns:
            bytes: Bytearray hasil konversi term frequencies.
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Mendekode list term frequencies dari aliran byte.

        Args:
            encoded_tf_list (bytes): Bytearray hasil dari metode encode_tf.

        Returns:
            list: List of term frequencies hasil dekoding.
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Kelas untuk kompresi postings list menggunakan algoritma Variable-Byte Encoding (VBE).
    Berbeda dengan StandardPostings, kelas ini menyimpan gap antar docID (delta encoding),
    kecuali untuk elemen pertama.

    Contoh:
    postings list [34, 67, 89, 454] diubah menjadi gap-based [34, 33, 22, 365],
    kemudian di-encode menggunakan algoritma Variable-Byte Encoding.

    Asumsi: postings_list untuk sebuah term muat di memori.
    """

    @staticmethod
    def vb_encode_number(number):
        """
        Mengonversi sebuah angka menjadi urutan byte menggunakan Variable-Byte Encoding.
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128)
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding terhadap list angka menggunakan Variable-Byte Encoding.
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Mengonversi postings_list menjadi aliran byte (VBE).
        DocID diubah menjadi bentuk gap-based (delta encoding) sebelum di-encode.

        Args:
            postings_list (list): List dari docID (postings).

        Returns:
            bytes: Bytearray hasil kompresi.
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Mengonversi list term frequencies menjadi aliran byte menggunakan VBE.

        Args:
            tf_list (list): List of term frequencies.

        Returns:
            bytes: Bytearray hasil kompresi TF.
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Mendekode aliran byte yang menggunakan Variable-Byte Encoding menjadi list angka.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Mendekode postings_list dari aliran byte (VBE).
        Hasil dekoding awal yang berupa gap-based akan dikonversi kembali menjadi docID asli.

        Args:
            encoded_postings_list (bytes): Bytearray hasil kompresi VBE.

        Returns:
            list: List of docID asli hasil dekoding.
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Mendekode list term frequencies dari aliran byte menggunakan VBE.

        Args:
            encoded_tf_list (bytes): Bytearray hasil kompresi VBE untuk TF.

        Returns:
            list: List of term frequencies hasil dekoding.
        """
        return VBEPostings.vb_decode(encoded_tf_list)

class OptPForDeltaPostings:
    BLOCK_SIZE = 128

    """ 
    Kelas untuk kompresi postings list menggunakan algoritma OptPForDelta.
    Algoritma ini membagi list menjadi blok-blok berukuran tetap (128 elemen),
    menentukan bit-width (b) yang optimal untuk setiap blok, dan memisahkan
    pengecualian (exceptions) yang di-encode secara terpisah.

    Asumsi:
    1. postings_list muat di memori.
    2. Ukuran blok tetap sebesar 128 elemen.
    """

    @staticmethod
    def encode_opt_block(block):
        """
        Mengompresi satu blok menggunakan OptPForDelta.

        Args:
            block (list): List of gaps atau angka dalam satu blok.

        Returns:
            bytes: Bytearray hasil kompresi satu blok.
        """
        sorted_block = sorted(block)
        b = sorted_block[int(0.9 * (len(sorted_block) - 1))].bit_length()

        main_data = []
        outliers_val = []
        outliers_idx = []
        for i, d in enumerate(block):
            if d < (1 << b):
                main_data.append(d)
            else:
                main_data.append(0)
                outliers_val.append(d)
                outliers_idx.append(i)

        header = array.array('B')
        header.append(b & 0xFF)
        header.extend(len(block).to_bytes(1, 'big'))
        header.extend(len(outliers_val).to_bytes(1, 'big'))

        packed_data = array.array('B')
        current, bits = 0, 0
        for val in main_data:
            current = (current << b) | (val & ((1 << b) - 1))
            bits += b
            while bits >= 8:
                bits -= 8
                packed_data.append((current >> bits) & 0xFF)     
                current &= (1 << bits) - 1  
        if bits > 0:
            packed_data.append((current << (8 - bits)) & 0xFF)

        outlier_stream = VBEPostings.vb_encode(outliers_val) + VBEPostings.vb_encode(outliers_idx)

        return header.tobytes() + packed_data.tobytes() + outlier_stream

    @staticmethod
    def encode_opt(postings_list):
        """
        Mengompresi list angka ke dalam aliran byte menggunakan OptPForDelta per blok.

        Args:
            postings_list (list): List of numbers (gaps atau TF).

        Returns:
            bytes: Aliran byte hasil kompresi seluruh list.
        """
        N = OptPForDeltaPostings.BLOCK_SIZE
        full_steam = bytearray()

        total_postings = len(postings_list)
        full_steam.extend(total_postings.to_bytes(4, 'big'))

        for i in range(0, total_postings, N):
            block = postings_list[i:min(i + N, total_postings)]
            
            block_bytes = OptPForDeltaPostings.encode_opt_block(block)

            block_len = len(block_bytes)
            full_steam.extend(block_len.to_bytes(4, 'big'))

            full_steam.extend(block_bytes)
        
        return bytes(full_steam)

    @staticmethod
    def encode(postings_list):
        """
        Mengompresi postings_list menggunakan OptPForDelta.
        DocID akan diubah menjadi gap-based sebelum dikompresi.

        Args:
            postings_list (list): List of docID (postings).

        Returns:
            bytes: Aliran byte hasil kompresi OptPForDelta.
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])

        return OptPForDeltaPostings.encode_opt(gap_postings_list)

    @staticmethod
    def decode_opt_block(block_bytes):
        """
        Mendekode satu blok hasil kompresi OptPForDelta.

        Args:
            block_bytes (bytes): Bytearray kompresi per blok.

        Returns:
            list: List angka yang telah didekode.
        """
        if not block_bytes:
            return []
        
        b = block_bytes[0]
        N = block_bytes[1]
        num_exceptions = block_bytes[2]
        
        result = []
        ptr = 3
        curr_val, bits_left = 0, 0
        for _ in range(N):
            val, needed = 0, b
            while needed > 0:
                if bits_left == 0:
                    curr_val = block_bytes[ptr]
                    ptr += 1
                    bits_left = 8
                
                take = min(needed, bits_left)
                val = (val << take) | ((curr_val >> (bits_left - take)) & ((1 << take) - 1))
                bits_left -= take
                needed -= take
            
            result.append(val)

        outliers = VBEPostings.vb_decode(block_bytes[ptr:])
        for i in range(num_exceptions):
            result[outliers[i + num_exceptions]] = outliers[i]
        
        return result

    @staticmethod
    def decode_opt(encoded_postings_list):
        """
        Mendekode seluruh aliran byte OptPForDelta kembali menjadi list angka.

        Args:
            encoded_postings_list (bytes): Aliran byte hasil kompresi.

        Returns:
            list: List angka asli (masih dalam bentuk gaps atau TF).
        """
        stream = array.array('B')
        stream.frombytes(encoded_postings_list)
        
        total_postings = int.from_bytes(stream[0:4], 'big')

        ptr = 4
        decoded_gap_postings_list = []
        
        while len(decoded_gap_postings_list) < total_postings:
            block_len = int.from_bytes(stream[ptr : ptr + 4], 'big')
            ptr += 4

            block_bytes = stream[ptr : ptr + block_len]
            ptr += block_len

            decoded_block = OptPForDeltaPostings.decode_opt_block(block_bytes)

            decoded_gap_postings_list.extend(decoded_block)

        return decoded_gap_postings_list

    @staticmethod
    def decode(encoded_postings_list):
        """
        Mendekode aliran byte OptPForDelta kembali menjadi list docID asli.
        Mengonversi kembali dari gap-based menjadi nilai asli (cumulative sum).

        Args:
            encoded_postings_list (bytes): Aliran byte hasil kompresi.

        Returns:
            list: List of docID asli.
        """
        decoded_gap_postings_list = OptPForDeltaPostings.decode_opt(encoded_postings_list)
        
        postings_list = [decoded_gap_postings_list[0]]
        for i in range(1, len(decoded_gap_postings_list)):
            postings_list.append(decoded_gap_postings_list[i] + postings_list[i-1])
        
        return postings_list

    @staticmethod
    def encode_tf(tf_list):
        """
        Mengompresi list term frequencies menggunakan OptPForDelta.

        Args:
            tf_list (list): List of term frequencies.

        Returns:
            bytes: Aliran byte hasil kompresi.
        """
        return OptPForDeltaPostings.encode_opt(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Mendekode aliran byte hasil kompresi TF (OptPForDelta).

        Args:
            encoded_tf_list (bytes): Aliran byte hasil kompresi TF.

        Returns:
            list: List of term frequencies hasil dekoding.
        """
        return OptPForDeltaPostings.decode_opt(encoded_tf_list)

if __name__ == '__main__':
    
    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, OptPForDeltaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("byte hasil encode postings: ", encoded_postings_list)
        print("ukuran encoded postings   : ", len(encoded_postings_list), "bytes")
        print("byte hasil encode TF list : ", encoded_tf_list)
        print("ukuran encoded postings   : ", len(encoded_tf_list), "bytes")
        
        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("hasil decoding (postings): ", decoded_posting_list)
        print("hasil decoding (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original"
        assert decoded_tf_list == tf_list, "hasil decoding tidak sama dengan postings original"
        print()

    print(" --- Big Test Case (500 elements) --- ")
    
    big_postings_list = []
    current_id = random.randint(1, 100)
    for _ in range(500):
        big_postings_list.append(current_id)
        current_id += random.randint(1, 100)

    big_postings_list[-2] = big_postings_list[-3] + random.randint(2 ** 10, 2 ** 11 - 1)
    big_postings_list[-1] = big_postings_list[-2] + random.randint(2 ** 20, 2 ** 21 - 1)
    
    big_tf_list = [random.randint(1, 100) for _ in range(500)]
    
    for Postings in [StandardPostings, VBEPostings, OptPForDeltaPostings]:
        encoded_postings = Postings.encode(big_postings_list)
        encoded_tf = Postings.encode_tf(big_tf_list)
        
        print(f"{Postings.__name__}:")
        print(f"  Postings encoded size: {len(encoded_postings)} bytes")
        print(f"  TF list encoded size : {len(encoded_tf)} bytes")
        
        assert Postings.decode(encoded_postings) == big_postings_list
        assert Postings.decode_tf(encoded_tf) == big_tf_list
