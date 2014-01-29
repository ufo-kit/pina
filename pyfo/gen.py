import parser
import qualifiers
import pyfo.opt
import pyfo.cast
from pycparser import c_generator, c_ast


def fix_signature(fdef, specs):
    """Add necessary qualifiers to the function signature."""
    params = [p for p in fdef.decl.type.args.params if p.name in specs]

    for p in params:
        spec = specs[p.name]

        if isinstance(spec.qualifier, qualifiers.NoQualifier):
            d = pyfo.cast.TypeDecl(p.name, 'float', None)
        else:
            d = pyfo.cast.PtrDecl(p.name, spec.qualifier.cl_keyword + ' float', None)

        pyfo.cast.replace(fdef.decl, p, d)


def fix_for_loops(fdef, specs):
    """Instantiate a real for loop now that we know sizes of data."""
    loops = [l for l in pyfo.cast.find_type(fdef.body, c_ast.For) if hasattr(l, '_extra')]

    for loop in loops:
        it, mem = loop._extra

        if mem in specs:
            it_var = c_ast.ID(it + '__it')
            n_it = specs[mem].size / 4
            loop.init = pyfo.cast.TypeDecl(it_var.name, 'int', c_ast.Constant('int', '0'))
            loop.cond = c_ast.BinaryOp('<', it_var, c_ast.Constant('int', str(n_it)))

            update = c_ast.ExprList([c_ast.BinaryOp('+=', it_var, c_ast.Constant('int', '1')),
                                     c_ast.Assignment('=', c_ast.ID(it), pyfo.cast.ArrayRef(mem, it_var.name))])
            loop.next = update
        else:
            raise TypeError("Cannot infer iterator type")


def replace_global_accesses(fdef, specs):
    """Replace all reads and writes on global variabls with array accesses."""
    names = [n for n in pyfo.cast.find_global_names(fdef)
             if n in specs and not isinstance(specs[n].qualifier, qualifiers.NoQualifier)]

    for name in names:
        # Replace simple identifiers
        for node in pyfo.cast.find_name(fdef.body, name):
            read_access = pyfo.cast.ArrayRef(name, 'idx')
            pyfo.cast.replace(fdef.body, node, read_access)

        # Replace already indexed accesses
        def is_valid(node):
            return isinstance(node, c_ast.ArrayRef) and node.name.name == name

        def is_constant_subscript(node):
            return is_valid(node) and isinstance(node.subscript, c_ast.Constant)

        def is_unary_op_subscript(node):
            return is_valid(node) and isinstance(node.subscript, c_ast.UnaryOp)

        for node in pyfo.cast.find(fdef.body, is_constant_subscript):
            node.subscript = c_ast.BinaryOp('+', c_ast.ID('idx'), node.subscript)

        for node in pyfo.cast.find(fdef.body, is_unary_op_subscript):
            subscript = node.subscript
            node.subscript = c_ast.BinaryOp(subscript.op, c_ast.ID('idx'), subscript.expr)


def fix_local_accesses(fdef):
    """Add a declaration for all referenced local variables"""
    localvars = []
    globalvars = list(pyfo.cast.find_global_names(fdef))
    assignments = pyfo.cast.find_type(fdef.body, c_ast.Assignment)

    for each in assignments:
        name = each.lvalue.name
        if not name in localvars and not name in globalvars:
            localvars.append(each.lvalue.name)

    for var in localvars:
        fdef.body.block_items.insert(0, pyfo.cast.TypeDecl(var, 'float', None))

    # create work item indices
    fdef.body.block_items.insert(0, pyfo.cast.TypeDecl('idx', 'int', pyfo.cast.WorkItemIndex()))


def replace_return_statements(fdef):
    """Turn all return statements into writes to a global 'out' buffer."""
    lvalue = pyfo.cast.ArrayRef('out', 'idx')

    for stmt in pyfo.cast.find_type(fdef.body, c_ast.Return):
        assignment = c_ast.Assignment('=', pyfo.cast.ArrayRef('out', 'idx'), stmt.expr)
        pyfo.cast.replace(fdef.body, stmt, assignment)

    # add out argument
    fdef.decl.type.args.params.append(pyfo.cast.PtrDecl('out', '__global float', None))


def replace_constants(fdef):
    consts = {
        'e':    '2.7182818284590452353602874713526624977572470937000',
        'ln2':  '0.6931471805599453094172321214581765680755001343602',
        'ln10': '2.3025850929940456840179914546843642076011014886288',
        'pi':   '3.1415926535897932384626433832795028841971693993751',
        'pi_2': '1.5707963267948966192313216916397514420985846996876',
    }

    def is_constant(node):
        return isinstance(node, c_ast.Constant) and node.value in consts

    for node in pyfo.cast.find(fdef.body, is_constant):
        pyfo.cast.replace(fdef.body, node, c_ast.ID(consts[node.value]))


def ast(func, specs, env=None):
    fdef = parser.parse(func)

    fix_signature(fdef, specs)
    fix_local_accesses(fdef)
    fix_for_loops(fdef, specs)
    replace_global_accesses(fdef, specs)
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
