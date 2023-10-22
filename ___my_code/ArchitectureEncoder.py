class ArchitectureEncoder:
    """
    Base class from which all architecture encoders should inherit.

    Only `method convert_from_standart_encoding` needs to be implemented.
    """

    def __init__(self) -> None:

        # Dictionary that maps all possible operation strings we can find
        # in NasBench201 to it's number. Taken from code from Paper 'Zero-Cost Proxies for Lightweight NAS'
        self._opname_to_num = {
            'none': 0,
            'skip_connect': 1,
            'nor_conv_1x1': 2,
            'nor_conv_3x3': 3,
            'avg_pool_3x3': 4
        }

        # Inverse dict of the `self._opname_to_num`
        self._num_to_opname = {
            v: k for (k, v) in self._opname_to_num.items()
        }
        
    def arch_str_to_standart_encoding(self, arch_str: int) -> list[int]:
        """
        Converts architecture string to Standart Encoding.

        Standart encoding is a vector of 6 integers. Each position of such vector represents
        operation of one edge in NasBench 201 cell. These integers range from 0 to 4 (including both).
        As NasBench 201 cell only has exactly 6 edges and each edge has one of 5 operations, this
        encoding can fully describe every cell from NasBench 201.

        Values at indices mean:
            0 - op from input    to 1st node

            1 - op from input    to 2nd node
            2 - op from 1st node to 2nd node

            3 - op from input    to 3rd node
            4 - op from 1st node to 3rd node
            5 - op from 2nd node to 3rd node

        This encoding and is from code from Paper 'Zero-Cost Proxies for Lightweight NAS'.

        Parameters
        ----------
        arch_str : str
            String describing the architecture as used in NasBench 201. 
            e.g. |avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|

        Returns
        -------
        standart_encoding : list[int]
            Standart encoding of supplied architecture string
        """
        edges = arch_str.split('+')
        edges = [edge[1:-1].split('|') for edge in edges]

        # Flatten list of edges       
        edges = [item for sublist in edges for item in sublist]

        # Remove '~n' parts
        edges = [edge.split('~')[0] for edge in edges]

        standart_encoding = [self._opname_to_num[edge] for edge in edges]
        return standart_encoding
    
    def standart_encoding_to_arch_str(self, std_encoding: list[int]) -> str:
        """
        Converts standart encoding vector to architecture string as specified by NasBench 201
        """

        # Go from list of operation indices to list of operation names
        std_encoding_opnames = [self._num_to_opname[encoded_op] for encoded_op in std_encoding]
        
        to_1st_node = f'|{std_encoding_opnames[0]}~0|'
        to_2nd_node = f'|{std_encoding_opnames[1]}~0|{std_encoding_opnames[2]}~1|'
        to_3rd_node = f'|{std_encoding_opnames[3]}~0|{std_encoding_opnames[4]}~1|{std_encoding_opnames[5]}~2|'

        arch_str = f'{to_1st_node}+{to_2nd_node}+{to_3rd_node}'

        return arch_str


    def convert_from_standart_encoding(self, std_encoding: list[int]) -> list[int]:
        """
        Returns encoding of a architecture defined by provided standart encoding.
        """
        raise NotImplementedError()

    def convert_from_arch_string(self, arch_str: str) -> list[int]:
        """
        Returns encoding of a architecture defined by provided NasBench 201 architecture string.
        """
        return self.convert_from_standart_encoding(self.arch_str_to_standart_encoding(arch_str))