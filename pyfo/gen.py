import parser
import qualifiers
import pyfo.opt
import pyfo.cast
from pycparser import c_generator, c_ast


def work_item_index(dims=2):
    """Build an expression to build the current work item index."""
    left = c_ast.ID('get_global_id(1)')
    right = c_ast.ID('get_global_size(0)')
    a = c_ast.BinaryOp('*', left, right)
    return c_ast.BinaryOp('+', a, c_ast.ID('get_global_id(0)'))


def array_ref(name, subscript):
    return c_ast.ArrayRef(c_ast.ID(name), c_ast.ID(subscript))


def decl(name, typename, init):
    idtype = c_ast.IdentifierType([typename])
    typedecl = c_ast.TypeDecl(name, [], idtype)
    return c_ast.Decl(name, None, None, None, typedecl, init, None)


def ptr_decl(name, typename, qualifiers):
    """Create a pointer declaration"""
    idtype = c_ast.IdentifierType([typename])
    typedecl = c_ast.TypeDecl(name, None, idtype)
    ptrdecl = c_ast.PtrDecl(None, typedecl)
    return c_ast.Decl(name, qualifiers, None, None, ptrdecl, None, None)


def local_decl(name, typename, init=None):
    typedecl =  c_ast.TypeDecl(name, [], c_ast.IdentifierType([typename]))
    return c_ast.Decl(name, None, None, None, typedecl, init, None)


def fix_signature(fdef, specs):
    """Add necessary qualifiers to the function signature."""
    params = [p for p in fdef.decl.type.args.params if p.name in specs]

    for p in params:
        spec = specs[p.name]

        if isinstance(spec.qualifier, qualifiers.NoQualifier):
            d = decl(p.name, 'float', None)
        else:
            d = ptr_decl(p.name, spec.qualifier.cl_keyword + ' float', None)

        pyfo.cast.replace(fdef.decl, p, d)


def fix_for_loops(fdef, specs):
    """Instantiate a real for loop now that we know sizes of data."""
    loops = [l for l in pyfo.cast.find_type(fdef.body, c_ast.For) if hasattr(l, '_extra')]

    for loop in loops:
        it, mem = loop._extra

        if mem in specs:
            it_var = c_ast.ID(it + '__it')
            n_it = specs[mem].size / 4
            loop.init = decl(it_var.name, 'int', c_ast.Constant('int', '0'))
            loop.cond = c_ast.BinaryOp('<', it_var, c_ast.Constant('int', str(n_it)))

            update = c_ast.ExprList([c_ast.BinaryOp('+=', it_var, c_ast.Constant('int', '1')),
                                     c_ast.Assignment('=', c_ast.ID(it), array_ref(mem, it_var.name))])
            loop.next = update
        else:
            raise TypeError("Cannot infer iterator type")


def global_vars(fdef):
    return (p.name for p in fdef.decl.type.args.params)


def replace_global_accesses(fdef):
    """Replace all reads and writes on global variabls with array accesses."""

    for name in global_vars(fdef):
        for node in pyfo.cast.find_name(fdef.body, name):
            read_access = array_ref(name, 'idx')
            pyfo.cast.replace(fdef.body, node, read_access)


def fix_local_accesses(fdef):
    """Add a declaration for all referenced local variables"""
    localvars = []
    globalvars = list(global_vars(fdef))
    assignments = pyfo.cast.find_type(fdef.body, c_ast.Assignment)

    for each in assignments:
        name = each.lvalue.name
        if not name in localvars and not name in globalvars:
            localvars.append(each.lvalue.name)

    for var in localvars:
        fdef.body.block_items.insert(0, local_decl(var, 'float'))

    # create work item indices
    fdef.body.block_items.insert(0, local_decl('idx', 'int', work_item_index()))


def replace_return_statements(fdef):
    """Turn all return statements into writes to a global 'out' buffer."""
    lvalue = array_ref('out', 'idx')

    for stmt in pyfo.cast.find_type(fdef.body, c_ast.Return):
        assignment = c_ast.Assignment('=', array_ref('out', 'idx'), stmt.expr)
        pyfo.cast.replace(fdef.body, stmt, assignment)

    # add out argument
    fdef.decl.type.args.params.append(ptr_decl('out', 'float', ['__global']))


def replace_constants(fdef):
    consts = {
        'E': '2.7182818284590452353602874713526624977572470937000',
        'LN2': '0.69314718055994530941723212145817656807550013436026',
        'LN10': '2.3025850929940456840179914546843642076011014886288',
        'PI': '3.1415926535897932384626433832795028841971693993751',
        'PI_2': '1.5707963267948966192313216916397514420985846996876',
    }

    for symbol, value in consts.items():
        for node in pyfo.cast.find_name(fdef.body, symbol):
            pyfo.cast.replace(fdef.body, node, c_ast.ID(value))


def ast(func, specs, env=None):
    fdef = parser.parse(func)

    fix_signature(fdef, specs)
    fix_local_accesses(fdef)
    fix_for_loops(fdef, specs)
    replace_global_accesses(fdef)
    replace_return_statements(fdef)

    if env:
        pyfo.opt.level1(fdef, specs, env)
        pyfo.opt.level2(fdef, specs, env)

    # we replace constants after optimization passes, because the symbols might be
    # removed by the optimization
    replace_constants(fdef)
    return fdef


def kernel(func, specs, env=None):
    """Build OpenCL kernel source string from *func*"""
    generator = c_generator.CGenerator()
    return generator.visit(ast(func, specs, env))
