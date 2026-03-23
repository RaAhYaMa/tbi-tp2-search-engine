import array

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend ke depan
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # bit awal pada byte terakhir diganti 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
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
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
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
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)

class OptPForDeltaPostings:
    BLOCK_SIZE = 128

    """ 
    OptPForDelta compression.
    Algoritma ini membagi list of integers ke dalam blok-blok berukuran tetap (128),
    menentukan jumlah bit (b) yang optimal secara heuristik untuk setiap blok, 
    dan memisahkan nilai-nilai yang tidak muat ke dalam b-bit (exceptions).

    ASUMSI: 
    1. postings_list untuk sebuah term MUAT di memori!
    2. Ukuran blok (N) adalah 128.
    """

    @staticmethod
    def encode_opt_block(block):
        """
        Encode satu blok menggunakan OptPForDelta.

        Parameters
        ----------
        block: List[int]
            List of gaps

        Returns
        -------
        bytes
            bytearray hasil kompresi OptPForDelta
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
                main_data.append(0) # Placeholder
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
        Encode postings_list atau tf_list ke dalam stream of bytes menggunakan OptPForDelta.
        Method ini membagi list menjadi blok-blok berukuran tetap (BLOCK_SIZE).

        Parameters
        ----------
        postings_list: List[int]
            List of numbers (bisa berupa gaps atau raw TF)

        Returns
        -------
        bytes
            bytearray hasil kompresi seluruh list
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
        Encode postings_list menjadi stream of bytes menggunakan OptPForDelta.
        
        Langkah-langkah:
        1. Ubah docIDs menjadi gap-based (delta-encoding).
        2. Bagi list of gaps menjadi blok-blok (misal per 128 elemen).
        3. Untuk setiap blok:
           a. Cari nilai 'b' (bit-width) menggunakan heuristik quantile (90%).
           b. Tentukan elemen mana yang menjadi 'exception' (>= 2^b).
           c. Simpan b, jumlah elemen, jumlah exception, data b-bit, dan data exception (VBE).
        4. Gabungkan semua blok menjadi satu bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray hasil kompresi OptPForDelta
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])

        return OptPForDeltaPostings.encode_opt(gap_postings_list)

    @staticmethod
    def decode_opt_block(block_bytes):
        """
        Decode satu blok dari bytearray hasil encode_opt_block.

        Parameters
        ----------
        block_bytes: bytes
            bytearray yang merepresentasikan satu blok kompresi

        Returns
        -------
        List[int]
            List of decoded numbers dalam satu blok
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
        Decode seluruh bytearray hasil encode_opt kembali menjadi list of numbers.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray hasil kompresi encode_opt

        Returns
        -------
        List[int]
            List of numbers (masih berupa gaps atau raw TF)
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
        Decode encoded_postings_list dari stream of bytes kembali ke list of docIDs.

        Langkah-langkah:
        1. Baca stream of bytes blok demi blok.
        2. Untuk setiap blok:
           a. Baca header (nilai b, jumlah elemen N, dan jumlah exception).
           b. Unpack b-bit data sebanyak N elemen.
           c. Baca data exception dan 'patch' ke posisi yang sesuai di blok.
        3. Ubah list of gaps kembali menjadi list of docIDs asli (cumulative sum).

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray hasil encode OptPForDelta

        Returns
        -------
        List[int]
            List of original docIDs
        """
        decoded_gap_postings_list = OptPForDeltaPostings.decode_opt(encoded_postings_list)
        
        postings_list = [decoded_gap_postings_list[0]]
        for i in range(1, len(decoded_gap_postings_list)):
            postings_list.append(decoded_gap_postings_list[i] + postings_list[i-1])
        
        return postings_list

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes menggunakan OptPForDelta.
        
        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray hasil kompresi OptPForDelta
        """
        return OptPForDeltaPostings.encode_opt(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode stream of bytes menjadi list of term frequencies.

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray hasil encode_tf

        Returns
        -------
        List[int]
            List of term frequencies
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
