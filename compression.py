import array
import random

class StandardPostings:
    """ 
    Class with static methods to convert postings list representation
    from an integer list to a byte sequence (bytes) using the array library.
    This method does not perform additional compression (only raw bytes).

    Assumption: postings_list for a term fits in memory.
    """

    @staticmethod
    def encode(postings_list):
        """
        Converts postings_list into a stream of bytes.

        Args:
            postings_list (list): List of docIDs (postings).

        Returns:
            bytes: Bytearray representing the integer sequence in the postings_list.
        """
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list from a byte stream.

        Args:
            encoded_postings_list (bytes): Bytearray result from the encode method.

        Returns:
            list: List of docIDs from decoding.
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Converts the term frequencies list into a byte stream.

        Args:
            tf_list (list): List of term frequencies.

        Returns:
            bytes: Bytearray result of term frequencies conversion.
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes the term frequencies list from a byte stream.

        Args:
            encoded_tf_list (bytes): Bytearray result from the encode_tf method.

        Returns:
            list: List of term frequencies from decoding.
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Class for postings list compression using the Variable-Byte Encoding (VBE) algorithm.
    Unlike StandardPostings, this class stores the gap between docIDs (delta encoding),
    except for the first element.

    Example:
    postings list [34, 67, 89, 454] is converted to gap-based [34, 33, 22, 365],
    then encoded using the Variable-Byte Encoding algorithm.

    Assumption: postings_list for a term fits in memory.
    """

    @staticmethod
    def vb_encode_number(number):
        """
        Converts a number into a byte sequence using Variable-Byte Encoding.
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
        Performs encoding on a list of numbers using Variable-Byte Encoding.
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Converts postings_list into a byte stream (VBE).
        DocIDs are converted to gap-based (delta encoding) before encoding.

        Args:
            postings_list (list): List of docIDs (postings).

        Returns:
            bytes: Bytearray compression result.
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Converts the term frequencies list into a byte stream using VBE.

        Args:
            tf_list (list): List of term frequencies.

        Returns:
            bytes: Bytearray result of TF compression.
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decodes a byte stream using Variable-Byte Encoding into a list of numbers.
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
        Decodes postings_list from a byte stream (VBE).
        Initial decoding results in gap-based form will be converted back to the original docIDs.

        Args:
            encoded_postings_list (bytes): Bytearray result of VBE compression.

        Returns:
            list: List of original docIDs from decoding.
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
        Decodes the term frequencies list from a byte stream using VBE.

        Args:
            encoded_tf_list (bytes): Bytearray result of VBE compression for TF.

        Returns:
            list: List of term frequencies from decoding.
        """
        return VBEPostings.vb_decode(encoded_tf_list)

class OptPForDeltaPostings:
    BLOCK_SIZE = 128

    """ 
    Class for postings list compression using the OptPForDelta algorithm.
    This algorithm divides the list into fixed-size blocks (128 elements),
    determines the optimal bit-width (b) for each block, and separates
    exceptions which are encoded separately.

    Assumption:
    1. postings_list fits in memory.
    2. Fixed block size of 128 elements.
    """

    @staticmethod
    def encode_opt_block(block):
        """
        Compresses one block using OptPForDelta.

        Args:
            block (list): List of gaps or numbers in one block.

        Returns:
            bytes: Bytearray compression result of one block.
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
        Compresses a list of numbers into a byte stream using OptPForDelta per block.

        Args:
            postings_list (list): List of numbers (gaps or TF).

        Returns:
            bytes: Byte stream result of compressing the entire list.
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
        Compresses postings_list using OptPForDelta.
        DocIDs will be converted to gap-based before compression.

        Args:
            postings_list (list): List of docIDs (postings).

        Returns:
            bytes: Byte stream result of OptPForDelta compression.
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])

        return OptPForDeltaPostings.encode_opt(gap_postings_list)

    @staticmethod
    def decode_opt_block(block_bytes):
        """
        Decodes one block of OptPForDelta compression result.

        Args:
            block_bytes (bytes): Bytearray compression per block.

        Returns:
            list: List of decoded numbers.
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
        Decodes the entire OptPForDelta byte stream back into a list of numbers.

        Args:
            encoded_postings_list (bytes): Byte stream compression result.

        Returns:
            list: Original numbers list (still in gaps or TF form).
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
        Decodes the OptPForDelta byte stream back into the original docID list.
        Converts back from gap-based to the original values (cumulative sum).

        Args:
            encoded_postings_list (bytes): Byte stream compression result.

        Returns:
            list: List of original docIDs.
        """
        decoded_gap_postings_list = OptPForDeltaPostings.decode_opt(encoded_postings_list)
        
        postings_list = [decoded_gap_postings_list[0]]
        for i in range(1, len(decoded_gap_postings_list)):
            postings_list.append(decoded_gap_postings_list[i] + postings_list[i-1])
        
        return postings_list

    @staticmethod
    def encode_tf(tf_list):
        """
        Compresses the term frequencies list using OptPForDelta.

        Args:
            tf_list (list): List of term frequencies.

        Returns:
            bytes: Byte stream compression result.
        """
        return OptPForDeltaPostings.encode_opt(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes the byte stream result of TF compression (OptPForDelta).

        Args:
            encoded_tf_list (bytes): Byte stream result of TF compression.

        Returns:
            list: List of term frequencies from decoding.
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
        assert decoded_posting_list == postings_list, "decoding result does not match original postings"
        assert decoded_tf_list == tf_list, "decoding result does not match original postings"
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
