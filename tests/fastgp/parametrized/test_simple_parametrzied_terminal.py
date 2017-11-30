import pytest

import fastgp.parametrized.simple_parametrized_terminals as sp


class TestRangeOperationTerminal:

    def test_initialize(self):
        term = sp.RangeOperationTerminal()
        variable_type_indices = [(1, 2)]
        names = ['xtdlta' + str(x) for x in range(2)]
        with pytest.raises(ValueError):
            term.initialize_parameters(variable_type_indices, names)
        variable_type_indices = [(2, 0)]
        names = ['xtdlta' + str(x) for x in range(2)]
        with pytest.raises(ValueError):
            term.initialize_parameters(variable_type_indices, names)
        variable_type_indices = [(1, 3)]
        names = ['xtdlta' + str(x) for x in range(2)]
        term.initialize_parameters(variable_type_indices, names)
        variable_type_indices = [(1, 3), (7, 30)]
        names = ['xtdlta' + str(x) for x in range(2)] + ['ytdlta' + str(x) for x in range(7, 30)]
        term.initialize_parameters(variable_type_indices, names)

    def test_initialize_manually(self):
        term = sp.RangeOperationTerminal()
        variable_type_indices = [(0, 2)]
        names = ['xtdlta' + str(x) for x in range(2)]
        with pytest.raises(ValueError):
            term.initialize_parameters(variable_type_indices, names, operation='cat', begin_range_name='xtdlta0',
                                       end_range_name='xtdlta1')
        with pytest.raises(ValueError):
            term.initialize_parameters(variable_type_indices, names, operation='cat', begin_range_name='xtlta0',
                                       end_range_name='xtdlta1')
        with pytest.raises(ValueError):
            term.initialize_parameters(variable_type_indices, names, operation='cat', begin_range_name='xtdlta0',
                                       end_range_name='xtdlt1')
        variable_type_indices = [(0, 2)]
        names = ['xtdlta' + str(x) for x in range(2)]
        term.initialize_parameters(variable_type_indices, names, operation='max', begin_range_name='xtdlta0',
                                   end_range_name='xtdlta1')
        variable_type_indices = [(0, 7)]
        names = ['xtdlta' + str(x) for x in range(7)]
        term.initialize_parameters(variable_type_indices, names, operation='max', begin_range_name='xtdlta3',
                                   end_range_name='xtdlta4')
