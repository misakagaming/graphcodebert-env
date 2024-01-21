namespace GEN.ORT.YYY
{
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  //
  //                    Source Code Generated by
  //                           CA Gen 8.6
  //
  //    Copyright (c) 2024 CA Technologies. All rights reserved.
  //
  //    Name: IYY10111_PARENT_CREATE_S         Date: 2024/01/09
  //    Target OS:   CLR                       Time: 13:41:55
  //    Target DBMS: ODBC/ADO.NET              User: AliAl
  //    Access Method: <NONE>         
  //
  //    Generation options:
  //    Debug trace option not selected
  //    Data modeling constraint enforcement not selected
  //    Optimized import view initialization not selected
  //    Enforce default values with DBMS not selected
  //    Init unspecified optional fields to NULL not selected
  //
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // using Statements
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  using System;
  using com.ca.gen.vwrt;
  using com.ca.gen.vwrt.types;
  using com.ca.gen.vwrt.vdf;
  using com.ca.gen.csu.exception;
  
  using com.ca.gen.abrt;
  using com.ca.gen.abrt.functions;
  using com.ca.gen.abrt.cascade;
  using com.ca.gen.abrt.manager;
  using com.ca.gen.abrt.trace;
  using com.ca.gen.exits.common;
  using com.ca.gen.odc;
  using System.Data;
  using System.Collections;
  
  public class IYY10111 : ABBase
  {
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // IMPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    IYY10111_IA WIa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // EXPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    IYY10111_OA WOa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK IMPORT/EXPORT VIEWS CLASS VARIABLES
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    GEN.ORT.YYY.MYY10111_IA Myy10111Ia;
    GEN.ORT.YYY.MYY10111_OA Myy10111Oa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // MISC DECLARATIONS AND PROTOTYPES 
    //    FOLLOW AS NEEDED:             
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool func_0022020326_esc_flag;
    //       +->   IYY10111_PARENT_CREATE_S          01/09/2024  13:41
    //       !       IMPORTS:
    //       !         Work View imp_reference iyy1_server_data (Transient,
    //       !                     Mandatory, Import only)
    //       !           userid
    //       !           reference_id
    //       !         Entity View imp iyy1_parent (Transient, Mandatory,
    //       !                     Import only)
    //       !           pkey_attr_text
    //       !           psearch_attr_text
    //       !           pother_attr_text
    //       !           ptype_tkey_attr_text
    //       !       EXPORTS:
    //       !         Entity View exp iyy1_parent (Transient, Export only)
    //       !           pinstance_id
    //       !           preference_id
    //       !         Work View exp_error iyy1_component (Transient, Export
    //       !                     only)
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !
    //       !     PROCEDURE STATEMENTS
    //       !
    //     1 !  NOTE: 
    //     1 !  PURPOSE(CONTINUED)
    //     1 !  
    //     2 !  NOTE: 
    //     2 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     2 !  !!!!!!!!!!!!
    //     2 !  Review the Pre-Post Conditions and Return/Reason Codes.
    //     2 !  
    //     3 !  NOTE: 
    //     3 !  PRE-CONDITION
    //     3 !  All mandatory fields are given.
    //     3 !  POST-CONDITION
    //     3 !  Record is created.
    //     3 !  Return Code = 1, Reason Code = 1
    //     3 !  
    //     4 !  NOTE: 
    //     4 !  PRE-CONDITION
    //     4 !  Record already exists.
    //     4 !  POST-CONDITION
    //     4 !  Record cannot be created.
    //     4 !  Return Code = -40, Reason Code = 2
    //     4 !  
    //     5 !  NOTE: 
    //     5 !  PRE-CONDITION
    //     5 !  At least one of the record fields is out of the permitted
    //     5 !  value boundaries.
    //     5 !  POST-CONDITION
    //     5 !  Record cannot be created.
    //     5 !  Return Code = -40, Reason Code = 3
    //     5 !  
    //     6 !  NOTE: 
    //     6 !  PRE-CONDITION
    //     6 !  TYPE could not be found.
    //     6 !  POST-CONDITION
    //     6 !  Record cannot be created.
    //     6 !  Return Code = -40, Reason Code = 5
    //     6 !  
    //     7 !  NOTE: 
    //     7 !  RETURN / REASON CODES
    //     7 !  +1/1 Record has been created.
    //     7 !  +1999/1 Other warnings.
    //     7 !  -40/2 Record already exists.
    //     7 !  -40/3 At least one of the record fields is out of the
    //     7 !  permitted value boundaries.
    //     7 !  -40/5 TYPE could not be found.
    //     7 !  -1999/1 Other errors.
    //     7 !  
    //     8 !  NOTE: 
    //     8 !  RELEASE HISTORY
    //     8 !  01_00 23-02-1998 New release
    //     8 !  
    //     9 !  NOTE: 
    //     9 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     9 !  !!!!!!!!!!!!
    //     9 !  USE <mapper ab>
    //     9 !  
    //    10 !  USE myy10111_parent_create
    //    10 !     WHICH IMPORTS: Entity View imp iyy1_parent TO Entity View
    //    10 !              imp iyy1_parent
    //    10 !                    Work View imp_reference iyy1_server_data TO
    //    10 !              Work View imp_reference iyy1_server_data
    //    10 !     WHICH EXPORTS: Entity View exp iyy1_parent FROM Entity View
    //    10 !              exp iyy1_parent
    //    10 !                    Work View exp_error iyy1_component FROM Work
    //    10 !              View exp_error iyy1_component
    //       +---
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //  CONSTRUCTOR FOR THE CLASS       
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    public IYY10111(  )
    {
      IefCGenRlse = "CA Gen 8.6";
      IsCopyright = "Copyright (c) 2024 CA Technologies. All rights reserved.";
      IefCGenDate = "2024/01/09";
      IefCGenTime = "13:41:55";
      IefCGenEncy = "9.2.A6";
      IefCGenUserId = "AliAl";
      IefCGenModel = "N8I_ORT_YYY_0112_TEMPLATE";
      IefCGenSubset = "ALL";
      IefCGenName = "IYY10111_PARENT_CREATE_S";
      NestingLevel = 0;
      ValChkDeadlockTimeout = false;
      ValChkDBError = false;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK FUNCTION DECLARATIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void Execute( Object in_runtime_parm1, 
    	IRuntimePStepContext in_runtime_parm2, 
    	GlobData in_globdata, 
    	IYY10111_IA import_view, 
    	IYY10111_OA export_view )
    {
      IefRuntimeParm1 = in_runtime_parm1;
      IefRuntimeParm2 = in_runtime_parm2;
      Globdata = in_globdata;
      WIa = import_view;
      WOa = export_view;
      _Execute();
    }
    
    private void _Execute()
    {
      
      ++(NestingLevel);
      try {
        f_22020326_init(  );
        f_22020326(  );
      } catch( Exception e ) {
        if ( ((Globdata.GetErrorData().GetStatus() == ErrorData.StatusNone) && (Globdata.GetErrorData().GetErrorEncountered() == 
          ErrorData.ErrorEncounteredNoErrorFound)) && (Globdata.GetErrorData().GetViewOverflow() == 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          Globdata.GetErrorData().SetStatus( ErrorData.LastStatusFatalError );
          Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusUnexpectedExceptionError );
          Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
          Globdata.GetErrorData(  ).SetErrorMessage( e );
        }
      }
      --(NestingLevel);
    }
    public void f_22020326(  )
    {
      func_0022020326_esc_flag = false;
      Globdata.GetStateData().SetCurrentABId( "0022020326" );
      Globdata.GetStateData().SetCurrentABName( "IYY10111_PARENT_CREATE_S" );
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PURPOSE(CONTINUED)                                              
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    Review the Pre-Post Conditions and Return/Reason Codes.         
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PRE-CONDITION                                                   
      //    All mandatory fields are given.                                 
      //    POST-CONDITION                                                  
      //    Record is created.                                              
      //    Return Code = 1, Reason Code = 1                                
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PRE-CONDITION                                                   
      //    Record already exists.                                          
      //    POST-CONDITION                                                  
      //    Record cannot be created.                                       
      //    Return Code = -40, Reason Code = 2                              
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PRE-CONDITION                                                   
      //    At least one of the record fields is out of the permitted       
      //    value boundaries.                                               
      //    POST-CONDITION                                                  
      //    Record cannot be created.                                       
      //    Return Code = -40, Reason Code = 3                              
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PRE-CONDITION                                                   
      //    TYPE could not be found.                                        
      //    POST-CONDITION                                                  
      //    Record cannot be created.                                       
      //    Return Code = -40, Reason Code = 5                              
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RETURN / REASON CODES                                           
      //    +1/1 Record has been created.                                   
      //    +1999/1 Other warnings.                                         
      //    -40/2 Record already exists.                                    
      //    -40/3 At least one of the record fields is out of the           
      //    permitted value boundaries.                                     
      //    -40/5 TYPE could not be found.                                  
      //    -1999/1 Other errors.                                           
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RELEASE HISTORY                                                 
      //    01_00 23-02-1998 New release                                    
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    USE <mapper ab>                                                 
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000010" );
      
      Myy10111Ia = (GEN.ORT.YYY.MYY10111_IA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.MYY10111).Assembly,
      	"GEN.ORT.YYY.MYY10111_IA" ));
      Myy10111Oa = (GEN.ORT.YYY.MYY10111_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.MYY10111).Assembly,
      	"GEN.ORT.YYY.MYY10111_OA" ));
      Myy10111Ia.ImpReferenceIyy1ServerDataUserid = FixedStringAttr.ValueOf(WIa.ImpReferenceIyy1ServerDataUserid, 8);
      Myy10111Ia.ImpReferenceIyy1ServerDataReferenceId = TimestampAttr.ValueOf(WIa.ImpReferenceIyy1ServerDataReferenceId);
      Myy10111Ia.ImpIyy1ParentPkeyAttrText = FixedStringAttr.ValueOf(WIa.ImpIyy1ParentPkeyAttrText, 5);
      Myy10111Ia.ImpIyy1ParentPsearchAttrText = FixedStringAttr.ValueOf(WIa.ImpIyy1ParentPsearchAttrText, 25);
      Myy10111Ia.ImpIyy1ParentPotherAttrText = FixedStringAttr.ValueOf(WIa.ImpIyy1ParentPotherAttrText, 25);
      Myy10111Ia.ImpIyy1ParentPtypeTkeyAttrText = FixedStringAttr.ValueOf(WIa.ImpIyy1ParentPtypeTkeyAttrText, 4);
      Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
      IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.MYY10111).Assembly,
      	"GEN.ORT.YYY.MYY10111",
      	"Execute",
      	Myy10111Ia,
      	Myy10111Oa );
      if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
        ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
        ErrorData.ErrorEncounteredNoErrorFound) )
      {
        throw new ABException();
      }
      Globdata.GetStateData().SetCurrentABId( "0022020326" );
      Globdata.GetStateData().SetCurrentABName( "IYY10111_PARENT_CREATE_S" );
      Globdata.GetStateData().SetLastStatementNumber( "0000000010" );
      WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Myy10111Oa.ExpErrorIyy1ComponentSeverityCode, 1);
      WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Myy10111Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
      WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Myy10111Oa.ExpErrorIyy1ComponentOriginServid);
      WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(Myy10111Oa.ExpErrorIyy1ComponentContextString, 512);
      WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Myy10111Oa.ExpErrorIyy1ComponentReturnCode);
      WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Myy10111Oa.ExpErrorIyy1ComponentReasonCode);
      WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Myy10111Oa.ExpErrorIyy1ComponentChecksum, 15);
      WOa.ExpIyy1ParentPinstanceId = TimestampAttr.ValueOf(Myy10111Oa.ExpIyy1ParentPinstanceId);
      WOa.ExpIyy1ParentPreferenceId = TimestampAttr.ValueOf(Myy10111Oa.ExpIyy1ParentPreferenceId);
      Myy10111Ia.FreeInstance(  );
      Myy10111Ia = null;
      Myy10111Oa.FreeInstance(  );
      Myy10111Oa = null;
      return;
    }
    
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // SUBORDINATE FUNCTIONS
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // INITIALIZATION UTILITY FUNCTIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void f_22020326_init(  )
    {
      if ( NestingLevel < 2 )
      {
      }
      WOa.ExpIyy1ParentPinstanceId = "00000000000000000000";
      WOa.ExpIyy1ParentPreferenceId = "00000000000000000000";
      WOa.ExpErrorIyy1ComponentSeverityCode = " ";
      WOa.ExpErrorIyy1ComponentRollbackIndicator = " ";
      WOa.ExpErrorIyy1ComponentOriginServid = 0.0;
      WOa.ExpErrorIyy1ComponentContextString = "";
      WOa.ExpErrorIyy1ComponentReturnCode = 0;
      WOa.ExpErrorIyy1ComponentReasonCode = 0;
      WOa.ExpErrorIyy1ComponentChecksum = "               ";
    }
  }// end class
  
}
