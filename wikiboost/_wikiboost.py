#! /usr/bin/env python

import os
import sys
import inspect
import re
import glob
import types
import importlib
import logging
import textwrap

from collections import namedtuple

import boost

from pyparsing import *

DefaultArg = namedtuple('DefaultArg', 'is_valid value')

DEF_LENGTH = 68
DEF_INDENT = 5
SPACE_ENC = "#'#[*.SPACE.*]#'#"
ELEM_OPEN = "#'#[*.ELEM_OPEN.*]#'#"
ELEM_CLOSE = "#'#[*.ELEM_CLOSE.*]#'#"

class MSIGError(Exception):
  pass

class HeaderError(MSIGError):
  pass

class InstanceError(MSIGError):
  pass

class CPPParserError(MSIGError):
  pass

class ParameterError(MSIGError):
  pass

class Notathing(object):
  pass

notathing = Notathing()

def _get_default_arg(args, defaults, i):
    if not defaults:
        return DefaultArg(False, None)

    args_with_no_defaults = len(args) - len(defaults)

    if i < args_with_no_defaults:
        return DefaultArg(False, None)
    else:
        value = defaults[i - args_with_no_defaults]
        if (type(value) is str):
            value = '"%s"' % value
        return DefaultArg(True, value)

def get_msig(method, space_encode=None):
    argspec = inspect.getargspec(method)
    i=0
    args = []
    for arg in argspec.args:
        default_arg = _get_default_arg(argspec.args, argspec.defaults, i)
        if default_arg.is_valid:
            val = default_arg.value
            if space_encode and isinstance(val, basestring):
              val = val.replace(" ", space_encode)
            args.append("%s=%s" % (arg, val))
        else:
            args.append(arg)
        i += 1
    return "%s(%s)" % (method.__name__, ", ".join(args))

def msig_as_source(method):
  rstr = ['<source lang="python">']
  rstr.append(get_msig(method))
  rstr.append('</source>')
  return "\n".join(rstr)

def formatted_msig_as_source(method, max_length=DEF_LENGTH,
                                     default_indent=DEF_INDENT):
  rstr = ['<source lang="python">']
  rstr.append(get_formatted_msig(method,
                                 max_length=max_length,
                                 default_indent=default_indent))
  rstr.append('</source>')
  return "\n".join(rstr) 

def format_msig(msig, space_encode, max_length=DEF_LENGTH,
                                    default_indent=DEF_INDENT):
  parts = msig.split(', ')
  longest_length = 0
  longest_index = 0
  for i, part in enumerate(parts):
    if len(part) > longest_length:
      longest_length = len(part)
      longest_index = i

  first_eq = None
  for i, part in enumerate(parts):
    if "=" in part:
      first_eq = i
      break

  joined = ", ".join(parts[:first_eq])
  replaced = joined.replace(space_encode, " ")
  indent1 = len(replaced) + 2
  joined = ", ".join(parts[:2])
  replaced = joined.replace(space_encode, " ")
  indent2 = len(replaced) + 2
  indent3 = parts[0].index("(") + 1
  indents = [(first_eq, indent1),
             (2, indent2),
             (1, indent3),
             (1, default_indent)]
  for idx, indent in indents:
    if idx is None:
      continue
    spaces = " " * indent
    lines = [", ".join(parts[:idx + 1])]
    for part in parts[idx + 1:]:
      part = spaces + part
      lines.append(part)
    lengths = [len(line) for line in lines]
    if max(lengths) < max_length:
      break
  return ",\n".join(lines).replace(space_encode, " ")

def get_formatted_msig(method, max_length=DEF_LENGTH,
                               default_indent=DEF_INDENT):
  msig = get_msig(method, space_encode=SPACE_ENC)
  return format_msig(msig, SPACE_ENC, max_length)

def wrap_source(source):
  return '<source lang="python">\n%s\n</source>' % source

def wrap_code(code):
  return "<code>%s</code>" % code

def wrap_html_element(s):
  return ELEM_OPEN + s + ELEM_CLOSE

def wrap_html_pre(pre, html_class=None):
  rstr = []
  if html_class is None:
    tag = "pre"
  else:
    tag = 'pre class="%s"' % html_class
  rstr.append(ELEM_OPEN + tag + ELEM_CLOSE)
  rstr.append(pre)
  rstr.append(ELEM_OPEN + "/pre" + ELEM_CLOSE)
  return "\n".join(rstr)

def entag_html(html):
  html = html.replace("<", "&lt;")
  html = html.replace(">", "&gt;")
  html = html.replace(ELEM_OPEN, "<")
  html = html.replace(ELEM_CLOSE, ">")
  return html


def collapse_code(code, title, html_class=None):
  elem = 'div class="toccolours mw-collapsible mw-collapsed'
  if html_class is not None:
    elem += " %s" % html_class
  elem += '"'
  elem1 = wrap_html_element(elem)
  elem = 'div class="mw-collapsible-content"'
  elem2 = wrap_html_element(elem)
  pre = wrap_html_pre(code)
  elem3 = wrap_html_element("/div")
  rstr = [elem1, title, elem2, pre, elem3, elem3]
  return "\n".join(rstr)

def get_class_ref(obj):
  if isinstance(obj, (types.ClassType, boost.python.meta_class)):
    class_repr = repr(obj)
  else:
    class_repr = repr(type(obj))
  try:
    class_ref = clsrepr.parseString(class_repr)
  except ParseException:
    msg = "%s does not seem to be instance of a boost class." % (obj,)
    raise InstanceError(msg)
  return class_ref

def chunk_cppsig(s):
  lines = s.splitlines()
  sigs = []
  for i, line in enumerate(lines):
    if "C++ signature :" in line:
      sigs.append(lines[i+1])
  return sigs


def wikify(obj, methods, config, module=None):
  logging.info("Object: %s", obj)
  rstr = ["{{TOClimit|2}}\n"]
  for m, msig in methods:
    logging.info("Method: '%s'", m)
    docstring = None
    rstr.append('<div class="method-doc">')
    rstr.append("== %s() ==\n" % m)
    rstr.append('<div class="method-usage">')
    rstr.append("=== usage: %s ===\n" % m)
    f = getattr(obj, m)
    sig_in_docstring = False
    if f.__doc__ is not None:
      docstring = f.__doc__.strip()
      if docstring:
        ds = docstring.splitlines()
        nlines = len(ds)
        for i in xrange(nlines):
           s = "\n".join(ds[i:])
           try:
             csig = call_sigs.parseString(s)
           except ParseException as e:
             for aline in chunk_cppsig(s):
               parse_cppsig(aline, m)
             continue
           msig = []
           for sig in ds[i+3::5]:
              msig.append(sig.strip())
           sig_in_docstring = True
           j = i + (len(msig) * 5) - 1
           start = sum((len(lin) + 1) for lin in ds[:i])
           end = start + sum((len(lin) + 1) for lin in ds[i:j]) - 1
           doc_sig_idx = (start, end)
           break
      else:
        docstring = None
    if msig is None:
      # it's a python method
      msig = formatted_msig_as_source(f)
      rstr.append(msig)
      rstr.append("</div> <!-- method-usage -->")
      rstr.append("")
    else:
      # it's a boost method so msig (should be) a list
      if sig_in_docstring:
        start, end = doc_sig_idx
        to_collapse = docstring[start:end]
        title = "'''boost python docstring'''"
        collapsed = collapse_code(to_collapse, title,
                                  html_class='boost-autodoc')
        docstring = docstring[:start] + collapsed + docstring[end:]
      else:
        if isinstance(msig, basestring):
          msig = [msig]
        else:
          tplt = "Signature for '%s' of '%s' is not a string."
          msg = tplt % (m, obj)
          raise MSIGError(msg)
      if (msig is not None) and (not isinstance(msig, list)):
        msg = "Could not get signature for '%s' of '%s'." % (m, obj) 
        msg += "\n%s" % msig
        raise MSIGError(msg)
      for sig_idx, sig in enumerate(msig):
        sig_list = parse_cppsig(sig, m)
        if sig_in_docstring:
          return_type = sig_list[0]
        else:
          return_type = "??"
        arg_types = []
        opt_indices = []
        i = 0
        for arg in sig_list[1:]:
          if arg == "[":
            opt_indices.append(i)
          elif arg not in [",", "]"]:
            arg_types.append(arg)
            i += 1
        class_ref = get_class_ref(obj)
        if module is None:
          module = class_ref[:-1]
        cls = class_ref[-1]
        param_names = cpp_params(module, cls, m, sig_idx + 1, config)
        if not param_names:
          param_names = [("arg%s" % (i + 2)) for i in
                                               xrange(len(arg_types))]
        if m == "__init__":
          sig = ["%s(" % ".".join(class_ref)]
        else:
          sig = ["%s(" % m]
        for i, param_name in enumerate(param_names):
          if i == 0:
            if i in opt_indices:
              sig.append("[")
          else:
            if i in opt_indices:
              sig.append("[")
            sig.append(", ")
          sig.append(param_name)
        sig.append("]" * len(opt_indices))
        sig.append(")")
        sig = "".join(sig)
        sig = wrap_source(sig)
        rstr.append(sig)
        rstr.append("")
        if len(param_names) != len(arg_types):
          cls_full = ".".join(module + [cls, m])
          params = (cls_full, param_names, arg_types)
          tplt = ("Error counting parameters in %s.\n"
                  "  names: %s\n  types: %s")
          msg = tplt % params
          raise ParameterError(msg)
        if arg_types:
          type_list = ["'''Parameters'''"]
        else:
          type_list = ["<!-- '''Parameters''' -->"]
        for param_name, arg_type in zip(param_names, arg_types):
          type_list.append("*'''param:''' %s" % wrap_code(param_name))
          if "<code>" not in arg_type:
            arg_type = wrap_code(arg_type)
          type_list.append("** '''type:''' %s" %  arg_type)
        type_list ="\n".join(type_list)
        rstr.append(type_list)
        rstr.append("")
        if "<code>" not in return_type:
          return_type = wrap_code(return_type)
        returns = "'''Returns:''' %s" % return_type
        rstr.append(returns)
      rstr.append("</div> <!-- method-usage -->")
      rstr.append("")
    rstr.append('<div class="method-docstring">')
    rstr.append("=== documentation: %s ===\n" % m)
    rstr.append("<!--#### Add wiki edits below this line. ####-->\n")
    rstr.append("<!--#### Add wiki edits above this line. ####-->\n")
    if docstring is not None:
      # first line was de-dented with strip, so must dedent all else
      docstring = docstring.splitlines()
      docstring_0 = docstring[0]
      docstring = "\n".join(docstring[1:])
      docstring = textwrap.dedent(docstring)
      docstring = "\n".join([docstring_0, docstring])
      docstring = entag_html(docstring)
      rstr.append(docstring)
    rstr.append("</div> <!-- method-docstring -->")
    rstr.append("</div> <!-- method-doc -->")
    rstr.append("")
  return "\n".join(rstr)

def get_accessors_methods(obj):
  stdout = sys.stdout
  dn = open(os.devnull, "w")
  accessors = []
  methods = []
  all_methods = []
  for att in sorted(dir(obj)):
     if (not att.startswith('_')) or (att == "__init__") :
       fn = getattr(obj, att)
       if callable(fn):
         try:
           argspec = inspect.getargspec(fn)
         except TypeError:
           functions = None
           try:
             sys.stdout = dn
             fn()
             functions = accessors
             e = None
           except Exception as e:
             sys.stdout = stdout
             functions = methods
             pass
           else:
             sys.stdout = stdout
             # ensure that it is a Boost function
             try:
               fn(notathing)
             except Exception as e:
               pass
           if e is None:
             raise Exception("Unexpected non-Boost Function: %s" % att)
           else:
             msig = str(e).splitlines()[-1]
             # sys.stderr.write(att + "\n")
             # sys.stderr.write(msig + "\n\n")
             functions.append((att, msig))
             all_methods.append((att, msig))
         else:
           if len(argspec.args) == 1:
             accessors.append((att, None))
           else:
             methods.append((att, None))
           all_methods.append((att, None))
  dn.close()
  return accessors, methods, all_methods

def get_accessors(obj):
  return get_accessors_methods(obj)[0]

def get_methods(obj):
  return get_accessors_methods(obj)[1]

def get_all_methods(obj):
  return get_accessors_methods(obj)[2]

def wikify_accessors(obj, config, module=None):
  methods = get_accessors(obj)
  return wikify(obj, methods, config, module)

def wikify_methods(obj, config, module=None):
  methods = get_methods(obj)
  return wikify(obj, methods, config, module)

def wikify_all_methods(obj, config, module=None):
  methods = get_all_methods(obj)
  return wikify(obj, methods, config, module)

def cpp_params(module, cls, method, num, config):
  """
  Returns a list of param names.

  Param `module` is the module as a sequence. For example
  if the module is "cctbx.miller", then `module` would
  be ``["cctbx", "miller"]``. Param `cls` is the class name
  (e.g. ``"binner"``) and param `method` is the method name
  (e.g. ``"get_i_bin"``.
  """
  root = config['source_root']
  class_re = re.compile(r'^\s*class\s*%s\b' % cls)
  method_re = re.compile(r'^\s*%s\s*\(' % method)
  accessor_re = re.compile(r'^\s*%s\s*\((.*?)\)' % method)
  # param_re = re.compile(r'const&\s*([a-z_]+[a-z_0-9]*)\b')
  param_re = re.compile(r'\b([a-z_]+[a-z_0-9]*)(?:,|\))')
  inline_param_re = re.compile(r'\b([a-z_]+[a-z_0-9]*)(?:,|$)')
  modpath = os.path.join(root, *module)
  header_pattern = os.path.join(modpath, "*.h")
  # cpp_pattern = os.path.join(modpath, "*.cpp")
  headers = []
  header_glob = glob.glob(header_pattern)
  for header in header_glob:
    with open(header) as f:
      for aline in f:
        if class_re.match(aline):
          headers.append(header)
          break
  if (len(headers) < 1):
    # failed to find header in best guess: try brut force search
    for (pth, dirs, files) in os.walk(root):
      for fname in files:
         if fname.endswith(".h"):
           header = os.path.join(pth, fname)
           if not header in header_glob:
             with open(header) as f:
               for aline in f:
                 if class_re.match(aline):
                   headers.append(header)
                   break
  args = []
  if len(headers) > 1:
    if config['require_params']:
      msg = "More than one header with 'class %s'." % cls
      raise HeaderError(msg)
  elif (len(headers) < 1):
    if config['require_params']:
      msg = "No headers with 'class %s'." % cls
      raise HeaderError(msg)
  else:
    header = headers[0]
    found = False
    match_num = 0
    with open(header) as f:
        for aline in f:
          if method_re.match(aline):
            m = accessor_re.match(aline)
            if m:
              match_num += 1
              if match_num == num:
                found = True
                g1 = m.group(1).strip()
                if g1:
                  fa = inline_param_re.findall(g1)
                  # if method == "get_i_bin":
                  #   raise Exception(method + (": %s" % fa))
                  args.extend(fa)
                break
              else:
                continue
            else:
              lines = []
              for aline in f:
                m = param_re.search(aline)
                if m:
                  match_num += 1
                  if match_num == num:
                    found = True
                    lines.append(aline)
                    if ")" not in aline:
                      for aline in f:
                        if param_re.search(aline):
                          lines.append(aline)
                          if ")" in aline:
                            break
                        else:
                          cls_full = ".".join(module + [cls, method])
                          msg = "C++ error in %s (%s)." % (cls_full, header)
                          raise CPPParserError(msg)
                    lines = " ".join(lines)
                    args.extend(param_re.findall(lines))
                  else:
                    break
                else:
                  break
              if found:
                break
              elif match_num < num:
                continue
              else:
                cls_full = ".".join(module + [cls, method])
                msg = "No parameters found for '%s'." % cls_full
                raise HeaderError(msg)
  # if not found:
  #   cls_full = ".".join(module + [cls, method])
  #   msg = "No proper method found for '%s'." % (cls_full)
  #   raise HeaderError(msg)
  return args

def ntype_action(s, l, t):
  return t[-1]

def modref_action(s, l, t):
  return ".".join(t[::2])

def clsref_action(s, l, t):
  return t[::2]

def clsrepr_action(s, l, t):
  return t[1:-1:]

def retary_action(s, l, t):
  lookup = {"scitbx.af.versa<cctbx.miller.index<int>," +
            "scitbx.af.flex_grid<scitbx.af.small<int,10ul>>>":
                           "cctbx_array_family_flex_ext.miller_index",
            "scitbx.af.const_ref<cctbx.miller.index<int>," +
            "scitbx.af.trivial_accessor>":
                           "cctbx_array_family_flex_ext.miller_index",
            "scitbx.af.versa<cctbx.miller.index<int>," +
            "scitbx.af.flex_grid<scitbx.af.small<long,10ul>>>":
                           "cctbx_array_family_flex_ext.miller_index",
            "scitbx.af.flex_grid<scitbx.af.small<int,10ul>>":
                           "scitbx_array_family_flex_ext.grid",
            "scitbx.af.flex_grid<scitbx.af.small<long,10ul>>":
                           "scitbx_array_family_flex_ext.grid",
            "scitbx.af.const_ref<std.complex<float>," +
            "scitbx.af.trivial_accessor>":
                           "scitbx.array_family.flex.complex_double",
            "scitbx.af.const_ref<bool,"
            "scitbx.af.flex_grid<scitbx.af.small<int,10ul>>>":
                           "scitbx.array_family.flex.bool",
            "iotbx.mtz.data_group<cctbx.hendrickson_lattman<double>>":
                           "iotbx_mtz_ext.hl_group",
            "iotbx.mtz.data_group<int>":
                           "iotbx_mtz_ext.integer_group",
            "iotbx.mtz.data_group<double>":
                           "iotbx_mtz_ext.real_group"}

  # must be synched with retval_action
  known_containers = ["scitbx.af.shared",
                      "scitbx.af.tiny",
                      "scitbx.af.small",
                      "cctbx.miller.index"]
  if t[0] not in known_containers:
    v = "".join(t)
    if v in lookup:
      t = [lookup[v]]
    else:
      msg = "Unknown container '%s'.\n%s" % (v, s)
      raise MSIGError(msg)
  return t

def retval_action(s, l ,t):
  lookup = {"scitbx.af.shared": "scitbx.array_family.flex",
            "scitbx.af.tiny": "tuple",
            "scitbx.af.small": "tuple",
            "cctbx.miller.index": "tuple",
            "iotbx.mtz.object": "iotbx_mtz_ext.object"}

  if t[-1] == "{lvalue}":
    t = t[:-1]
    t[0] = lookup.get(t[0], t[0])
  if len(t) > 1:  # is a modref
    token = lookup.get(t[0], t[0])
    tokens = [wrap_code(token)]
    tokens.append(" of %ss" % wrap_code(t[2]))
    if len(t) > 4:  # has a length
      tokens.insert(0, "%s-" % t[4])
    t = ["".join(tokens)]
  return t


def trivial_param_action(s, l, t):
  return "scitbx.array_family.flex.%s" % t[2]

def shared_type_action(s, l, t):
  lookup = {"cctbx::miller::index<int>": "cctbx.miller.indices"}
  return wrap_code(lookup[t[2]])

def object_pointer_action(s, l, t):
  return "C++ <code>object</code> (??)"

def python_object_action(s, l, t):
  return "Python <code>object</code> (??)"

def index_param_action(s, l, t):
  return "3-<code>tuple</code> of <code>int</code>s for hkl"

def params_action(s, l, t):
  return t[1::2]

def options_action(s, l, t):
  t = t[2::4]
  t[0] = "[" + t[0]
  t[-1] = t[-1] + "]"

def cpp_sig_action(s, l, t):
  return t[0:1] + t[4:-1]



#######################################################################
# Parsing Constants
#######################################################################
lowers = "abcdefghijklmnopqrstuvwxyz"
lowernums = "abcdefghijklmnopqrstuvwxyz0123456789"
open_par = Literal("(")
close_par = Literal(")")
comma = Literal(",")
star = Literal("*")

identifier = Word(alphas + "_", alphanums + "_")

modsep = Literal("::")
clssep = Literal(".")
modref = OneOrMore(identifier + modsep) + identifier
modref.setParseAction(modref_action)
clsref = OneOrMore(identifier + clssep) + identifier
clsref.setParseAction(clsref_action)
clsrepr = Literal("<class '") + clsref + Literal("'>")
clsrepr.setParseAction(clsrepr_action)

int_ = Keyword("int")
double = Keyword("double")
double.setParseAction(lambda s, l, t: "double")
unsigned = Keyword("unsigned")
unsigned.setParseAction(lambda s, l, t: "")
signed = Keyword("signed")
signed.setParseAction(lambda s, l, t: "")
tiny = Keyword("tiny")
tiny.setParseAction(lambda s, l, t: "int")
short = Keyword("short")
short.setParseAction(lambda s, l, t: "int")
long_ = Keyword("long")
long_.setParseAction(lambda s, l, t: "int")
unsigned_short = unsigned + short
unsigned_long = unsigned + long_
unsigned_int = unsigned + int_
signed_short = signed + short
signed_long = signed + long_
signed_int = signed + int_
integer = (unsigned_short | unsigned_long | unsigned_int |
           signed_short | unsigned_short | signed_int |
           short | long_ | tiny | int_)

float_ = double
number = integer | float_

bool_ = Keyword("bool")

ntype = integer | float_ | bool_
ntype.setParseAction(ntype_action)

void = Keyword("void")
void.setParseAction(lambda s, l, t: "None")

char = Keyword("char")

const = Keyword("const")

std_string = Literal("std::string")
std_string.setParseAction(lambda s, l, t: "str")

trivial_accessor = Keyword("scitbx::af::trivial_accessor")
trivial = (ntype | std_string) + comma + trivial_accessor

index_param = Literal("cctbx::miller::index<int>")
index_param.setParseAction(index_param_action)

trivial_param = modref + Literal("<") + trivial + Literal(">")
trivial_param.setParseAction(trivial_param_action)

shared_container = Literal("scitbx::af::shared")
shared_contained = Literal("cctbx::miller::index<int>")
shared_type = (shared_container + Literal("<") +
               shared_contained + Literal(">"))
shared_type.setParseAction(shared_type_action)

char_array = char + const + star
char_array.setParseAction(lambda s, l, t: "str")

object_pointer = Literal("_object*")
object_pointer.setParseAction(object_pointer_action)

python_object = Literal("boost::python::api::object")
python_object.setParseAction(python_object_action)

n_ul = Word(nums) + Literal("ul")

ary_contents = Forward()
ary_contents << (Literal("<") +
                    ((ntype + comma + modref + Optional(ary_contents)) |
                     (ntype + Optional(comma + n_ul)) |
                     (modref + Optional(ary_contents) +
                      Optional(comma + modref +
                               Optional(ary_contents)))) +
                 Literal(">"))
retary = modref + ary_contents
retary.setParseAction(retary_action)

retval = ((shared_type | retary | char_array |
           python_object | modref | ntype | char | void) +
          Optional(Literal("{lvalue}")))
retval.setParseAction(retval_action)

lvalue = (void | object_pointer |
          ((retary | modref) + Optional(Literal("{lvalue}"))))

lvalue.setParseAction(lambda s, l, t: wrap_code(t[0]))


param = (python_object | index_param | char_array |
         trivial_param  | shared_type | lvalue | ntype | char)
params = ZeroOrMore(comma + param)
params.setParseAction(params_action)

option_assignment = ((ntype + Literal("=") + identifier) |
                     (char_array + Literal("=") + sglQuotedString))
option_assignment.setParseAction(lambda s, l, t: "".join(t))


options = Forward()
options << (Literal("[") +
            OneOrMore((comma + option_assignment) |
                      (comma + ntype)) +
            Optional(options) +
            Literal("]"))
options.setParseAction(options_action)

cpp_sig = (Optional(retval) + identifier +
           open_par + Optional(lvalue + params) +
           Optional(options) + close_par)

argument = open_par + identifier + close_par + identifier

call_options = Forward()
call_options << Optional(Literal("[") +
                         OneOrMore(comma + argument +
                                   Optional(Literal("=") +
                                   (identifier | sglQuotedString))) +
                         call_options +
                         Literal("]"))
call_sig = (identifier + open_par +
            Optional(argument + ZeroOrMore(comma + argument) +
                     call_options) +
            close_par + Literal("->") + Or(identifier, clsref) +
            Literal(":") + Literal("C++ signature :") + cpp_sig)

call_sigs = OneOrMore(call_sig)

#######################################################################


def parse_cppsig(cppsig, att):
  attribute = Keyword(att)
  r = Optional(retval)
  def _a(s, l, t):
    if not t:
      t = [wrap_code("None")]
    return t
  r.setParseAction(_a)

  sig = (r + attribute + open_par +
         Optional(lvalue + params) +
         Optional(options) + close_par)

  sig.setParseAction(cpp_sig_action)
  try:
    parsed = sig.parseString(cppsig)
  except ParseException as e:
    pointer = " " * (e.col - 1) + "^"
    tmplt = "Unrecognized signature for '%s'.\n%s\n%s\n%s"
    msg = tmplt % (att, cppsig, pointer, e)
    raise MSIGError(msg)
  return sig.parseString(cppsig)

def asrt_eq(a, b):
  assert tuple(a) == tuple(b)

def test_identifier():
  t1 = identifier.parseString("dsfadfs_da4")
  t2 = identifier.parseString("_dsfadfs_da4")
  t3 = identifier.parseString("d4sfadfs_da4")
  t4 = identifier.parseString("______")
  asrt_eq(t1, ['dsfadfs_da4'])
  asrt_eq(t2, ['_dsfadfs_da4'])
  asrt_eq(t3, ['d4sfadfs_da4'])
  asrt_eq(t4, ['______'])

def test_modref():
  s = "scitbx::af::const_ref<double, scitbx::af::trivial_accessor>"
  t1 = modref.parseString(s)
  asrt_eq(t1, ['scitbx.af.const_ref'])

def test_unsigned_long():
  t1 = unsigned_long.parseString("unsigned long")
  asrt_eq(t1, ['', 'int'])

def test_integer():
  t1 = integer.parseString("unsigned short")
  t2 = integer.parseString("unsigned long")
  t3 = integer.parseString("long")
  t4 = integer.parseString("short")
  asrt_eq(t1, ['', 'int'])
  asrt_eq(t2, ['', 'int'])
  asrt_eq(t3,  ['int'])
  asrt_eq(t4,  ['int'])

def test_ntype():
  s = ("unsigned long get_i_bin(cctbx::miller::binning "
       "{lvalue},double)")
  t1 = ntype.parseString("double")
  t2 = ntype.parseString("bool")
  t3 = integer.parseString("unsigned long")
  t4 = integer.parseString("long")
  t5 = ntype.parseString(s)
  asrt_eq(t1, ['float'])
  asrt_eq(t2, ['bool'])
  asrt_eq(t3, ['', 'int'])
  asrt_eq(t4, ['int'])
  asrt_eq(t5, ['int'])

def test_trivial_param():
  s = "scitbx::af::const_ref<double, scitbx::af::trivial_accessor>"
  t1 = trivial_param.parseString(s)
  asrt_eq(t1, ['scitbx.array_family.flex.float'])

def test_shared_type():
  s = "scitbx::af::shared<cctbx::miller::index<int>>"
  t1 = shared_type.parseString(s)
  asrt_eq(t1, ["<code>cctbx.miller.indices</code>"])

def test_retval():
  s1 = ("scitbx::af::shared<double> "
        "interpolate(cctbx::miller::binner {lvalue},"
        "scitbx::af::const_ref<double, "
        "scitbx::af::trivial_accessor>,double)")
  s2 = ("unsigned long get_i_bin(cctbx::miller::binning "
        "{lvalue},double)")
  s3 = "scitbx::af::flex_grid<scitbx::af::small<long, 10ul> >"
  s4 = "iotbx::mtz::object {lvalue}"
  t1 = retval.parseString(s1)
  t2 = retval.parseString(s2)
  t3 = retval.parseString(s3)
  t4 = retval.parseString(s4)
  asrt_eq(t1, ['<code>scitbx.array_family.flex</code> of '
               '<code>double</code>s'])
  asrt_eq(t2, ['int'])
  asrt_eq(t3, ['scitbx_array_family_flex_ext.grid'])
  asrt_eq(t4, ['iotbx_mtz_ext.object'])

def test_lvalue():
  s = "cctbx::miller::binner {lvalue}"
  t1 = lvalue.parseString(s)
  asrt_eq(t1, ['<code>cctbx.miller.binner</code>'])

def test_params():
  s = (",scitbx::af::const_ref<double, "
       "scitbx::af::trivial_accessor>,double")
  t1 = params.parseString(s)
  asrt_eq(t1, ['scitbx.array_family.flex.float', 'float'])

def test_parse_cppsig():
  s1 = ("scitbx::af::shared<double> "
        "interpolate(cctbx::miller::binner "
        "{lvalue},scitbx::af::const_ref<double, "
        "scitbx::af::trivial_accessor>,double)")
  att1 = "interpolate"
  s2 = ("double bin_d_min(cctbx::miller::binning "
        "{lvalue},unsigned long)")
  att2 = "bin_d_min"
  s3 = ("unsigned long get_i_bin(cctbx::miller::binning "
        "{lvalue},double)")
  att3 = "get_i_bin"
  s4 = ("unsigned long get_i_bin(cctbx::miller::binning "
        "{lvalue},cctbx::miller::index<int>)")
  att4 = "get_i_bin"
  t1 = parse_cppsig(s1, att1)
  t2 = parse_cppsig(s2, att2)
  t3 = parse_cppsig(s3, att3)
  t4 = parse_cppsig(s4, att4)
  asrt_eq(t1, ['<code>scitbx.array_family.flex</code> of '
               '<code>float</code>s',
               'scitbx.array_family.flex.float', 'float'])
  asrt_eq(t2, ['float', 'int'])
  asrt_eq(t3, ['int', 'float'])

def test_call_sig():
  s1 = """get_i_bin( (binning)arg1, (float)arg2) -> int :

    C++ signature :
        unsigned long get_i_bin(cctbx::miller::binning {lvalue},double)"""
  s2 = """get_i_bin( (binning)arg1, (object)arg2) -> int :

    C++ signature :
        unsigned long get_i_bin(cctbx::miller::binning {lvalue},cctbx::miller::index<int>)"""
  s3 = """miller_indices( (binner)arg1) -> miller_index :

    C++ signature :
        scitbx::af::shared<cctbx::miller::index<int> > miller_indices(cctbx::miller::binner {lvalue})"""
  s4 = """range_all( (binning)arg1) -> object :

    C++ signature :
        boost::python::api::object range_all(cctbx::miller::binning)"""
  print
  print s1
  print "---"
  print call_sig.parseString(s1)
  print
  print s2
  print "---"
  print call_sig.parseString(s2)
  print
  print s3
  print "---"
  print call_sig.parseString(s3)
  print
  print s4
  print "---"
  print call_sig.parseString(s4)
  

def test_parsing():
  test_identifier()
  test_modref()
  test_unsigned_long()
  test_integer()
  test_ntype()
  test_trivial_param()
  test_shared_type()
  test_retval()
  test_lvalue()
  test_params()
  test_parse_cppsig()

def test_setup(config):
  import cctbx.miller
  from iotbx import mtz, pdb
  from scitbx.array_family import flex

  mtz_name = config['mtz_filename']
  mtz_file = mtz.object(mtz_filename)

  pdb_name = config['pdb_name']
  pdb_inp = pdb.input(file_name=pdb_name)
  structure = pdb_inp.xray_structure_simple()
  miller = structure.structure_factors(d_min=2.85).f_calc()
  miller_sub = miller[20000:20002]

  flex.random_generator.seed(82364)
  size = miller.size()
  rand_sel_1 = flex.random_bool(size, 0.5)
  rand_sel_2 = flex.random_bool(size, 0.5)
  miller_1 = miller.select(rand_sel_1).randomize_phases()
  miller_2 = miller.select(rand_sel_2).randomize_phases()
  rand_doub_1 = flex.random_double(miller_1.size(), 0.1) + 0.015
  rand_doub_2 = flex.random_double(miller_2.size(), 0.1) + 0.015
  sigmas_1 = rand_doub_1 * miller_1.amplitudes().data()
  sigmas_2 = rand_doub_2 * miller_2.amplitudes().data()
  miller_1.set_sigmas(sigmas_1)
  miller_2.set_sigmas(sigmas_2)
  miller_1.set_observation_type_xray_amplitude()
  miller_2.set_observation_type_xray_amplitude()
  miller_1.as_intensity_array().i_over_sig_i()
  miller_2.as_intensity_array().i_over_sig_i()

  binner = miller.setup_binner(n_bins=20)
  indices = miller.indices()

  mtch_indcs = miller_1.match_indices(miller_2)
  mset = miller.set()

  # doc = wikify_all_methods(cctbx.miller.binning, config)
  # doc = wikify_all_methods(type(mset), config)
  # doc = wikify_all_methods(type(binner), config)
  # doc = wikify_all_methods(type(miller), config)
  # doc = wikify_all_methods(type(miller.data()), config)
  # doc = wikify_all_methods(type(indices), config)
  # doc = wikify_all_methods(type(mtch_indcs), config,
  #                          module=["cctbx", "miller"])

  return (mtch_indcs, ["cctbx", "miller"])

def test_cpp_params():
  module = ["cctbx", "miller"]
  cls = "binner"
  method = "interpolate"
  method = "get_i_bin"
  method = "count"
  method = "bin_d_range"
  method = "array_indices"
  method = "bin_centers"
  method = "selection"
  config = {'source_root': "/opt/cctbx/cctbx_sources"}
  args = cpp_params(module, cls, method, config)
  for a in args:
    print a


def test_ary_contents():
  s1 = """<cctbx::miller::index<int>, scitbx::af::flex_grid<scitbx::af::small<long, 10ul> > >"""
  print ary_contents.parseString(s1)

def test_retary():
  s1 = """scitbx::af::versa<cctbx::miller::index<int>, scitbx::af::flex_grid<scitbx::af::small<long, 10ul> > >"""
  print retary.parseString(s1)
  s2 = """scitbx::af::const_ref<bool, scitbx::af::flex_grid<scitbx::af::small<long, 10ul> > >"""
  print retary.parseString(s2)

def find_modules(root):
  modules = []
  packages = []
  for (pth, dirs, files) in os.walk(root):
    if "__init__.py" in files:
      packages.append(pth)
    for file in files:
       if file.endswith(".py"):
         packages.append((pth, file))
    

def walk_boost(modules):
  imported = []
  for mod_name in modules:
    logging.debug("Trying to import '%s'.", mod_name)
    mod = importlib.import_module(mod_name)
    imported.append((mod_name, mod))
  for mod_name, mod in imported:
    print mod_name, ":", mod
    for att in dir(mod):
      obj = getattr(mod, att)
      print "    ", att, ":", type(obj)


def test_walk_boost():
  modules = ["boost_adaptbx", "cctbx", "gltbx", "iotbx",
             "libtbx", "mmtbx", "omptbx", "rstbx", "scitbx", "smtbx"]
  walk_boost(modules)

def main():
    def my_method1(first, second=42, third='something',
                          fourth=99, fifth='dsfadsfdsfds',
                          some_very_long_parameter_name_with_a_value='abcd',
                          and_this_one_has_a_terribly_long_name_with_a_long_value_with_spaces='dftsdt dsf ds fa'):
        pass

    def my_method2(first, second=42, third='something',
                          fourth=99, fifth='dsfadsfdsfds',
                          some_very_long_parameter_name_with_a_value='abcd'):
        pass

    def my_method3(first, second=42, third='something',
                          fourth=99, fifth='dsfadsfdsfds'):
        pass

    for max_length in (55, 68, 80):
      print
      print (" %s " % max_length).center(max_length, "=")
      for my_method in (my_method1, my_method2, my_method3):
        print
        print get_formatted_msig(my_method, max_length=max_length)
        print

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG)

  # test_parsing()
  # test_call_sig()
  # test_walk_boost()
  # test_retary()
  # test_ary_contents()
  # test_retval()

  config = {"source_root":
                "/usr/local/cctbx-svn/sources/cctbx_project"
            "mtz_filename": "toxd.mtz",
            "pdb_name": '1imh.pdb',
            "require_params": True}

  boost_object, module = test_setup(config)
  boost_type = type(boost_object)
  doc = wikify_all_methods(boost_type, config, module)
